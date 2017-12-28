"""
Author Matthew Aitchison

Date December 2017

Analysis a CPTV video file and extracts tracks for any moving objects.  These tracks are then saved out to a
HDF5 file for use in training a neural net.
"""

import pytz
import datetime
from collections import namedtuple
import os

import numpy as np
import cv2
import h5py
import scipy.ndimage

from ml_tools.tools import get_image_subsection
from ml_tools.tools import Rectangle
from ml_tools.trackdatabase import TrackDatabase

from cptv import CPTVReader

class Region(Rectangle):
    """ Region is a rectangle extended to support mass. """
    def __init__(self, topleft_x, topleft_y, width, height, mass=0, pixel_variance=0, id=0):
        super().__init__(topleft_x, topleft_y, width, height)
        # number of active pixels in region
        self.mass = mass
        # how much pixels in this region have changed since last frame
        self.pixel_variance = pixel_variance
        # an identifier for this region
        self.id = id

    def copy(self):
        return Region(self.x, self.y, self.width, self.height, self.mass, self.pixel_variance, self.id)

TrackMovementStatistics = namedtuple(
    'TrackMovementStatistics',
    'movement max_offset score average_mass median_mass duration delta_std'
)
TrackMovementStatistics.__new__.__defaults__ = (0,) * len(TrackMovementStatistics._fields)

class Track:
    """ Bounds of a tracked object over time. """

    # keeps track of which id number we are up to.
    _track_id = 1

    def __init__(self, id=None):
        """
        Creates a new Track.
        :param id: id number for track, if not specified is provided by an auto-incrementer
        """

        # used to uniquely identify the track
        if not id:
            self.id = self._track_id
            self._track_id += 1
        else:
            self.id = id
        # frame number this track starts at
        self.start_frame = 0
        # datetime this track starts
        self.start_time = None
        # our bounds over time
        self.bounds_history = []
        # number frames since we lost target.
        self.frames_since_target_seen = 0
        # our current estimated horizontal velocity
        self.vel_x = 0
        # our current estimated vertical velocity
        self.vel_y = 0

    def add_frame(self, bounds: Region):
        """
        Adds a new point in time bounds and mass to track
        :param bounds: new bounds region
        """
        self.bounds_history.append(bounds.copy())
        self.frames_since_target_seen = 0

        if len(self) >= 2:
            self.vel_x = self.bounds_history[-1].mid_x - self.bounds_history[-2].mid_x
            self.vel_y = self.bounds_history[-1].mid_y - self.bounds_history[-2].mid_y
        else:
            self.vel_x = self.vel_y = 0

    def get_stats(self):
        """
        Returns statistics for this track, including how much it moves, and a score indicating how likely it is
        that this is a good track.
        :return: a TrackMovementStatistics record
        """

        if len(self) <= 1:
            return TrackMovementStatistics()

        # get movement vectors
        mass_history = [int(bound.mass) for bound in self.bounds_history]
        variance_history = [bound.pixel_variance for bound in self.bounds_history]
        mid_x = [bound.mid_x for bound in self.bounds_history]
        mid_y = [bound.mid_y for bound in self.bounds_history]
        delta_x = [mid_x[0] - x for x in mid_x]
        delta_y = [mid_y[0] - y for y in mid_y]
        vel_x = [cur - prev for cur, prev in zip(mid_x[1:], mid_x[:-1])]
        vel_y = [cur - prev for cur, prev in zip(mid_y[1:], mid_y[:-1])]

        movement = sum((vx ** 2 + vy ** 2) ** 0.5 for vx, vy in zip(vel_x, vel_y))
        max_offset = max((dx ** 2 + dy ** 2) ** 0.5 for dx, dy in zip(delta_x, delta_y))

        # the standard deviation is calculated by averaging the per frame variances.
        # this ends up being slightly different as I'm using /n rather than /(n-1) but that
        # shouldn't make a big difference as n = width*height*frames which is large.
        delta_std = float(np.mean(variance_history)) ** 0.5

        movement_points = (movement ** 0.5) + max_offset
        delta_points = delta_std * 25.0
        score = movement_points + delta_points

        stats = TrackMovementStatistics(
            movement=movement,
            max_offset=max_offset,
            average_mass=float(np.mean(mass_history)),
            median_mass=float(np.median(mass_history)),
            duration=len(self) / 9.0,
            delta_std=delta_std,
            score=score
        )

        return stats

    def get_track_region_score(self, region: Region):
        """
        Calculates a score between this track and a region of interest.  Regions that are close the the expected
        location for this track are given high scores, as are regions of a similar size.
        """
        expected_x = int(self.bounds.mid_x + self.vel_x)
        expected_y = int(self.bounds.mid_y + self.vel_y)

        distance = ((region.mid_x - expected_x) ** 2 + (region.mid_y - expected_y) ** 2) ** 0.5

        # ratio of 1.0 = 20 points, ratio of 2.0 = 10 points, ratio of 3.0 = 0 points.
        # area is padded with 50 pixels so small regions don't change too much
        size_difference = (abs(region.area - self.bounds.area) / (self.bounds.area+50)) * 100

        return distance, size_difference

    @property
    def mass(self):
        return self.bounds_history[-1].mass

    @property
    def bounds(self) -> Region:
        return self.bounds_history[-1]

    @property
    def duration(self):
        """ Duration of track in seconds. """
        return len(self) / 9.0

    def __len__(self):
        return len(self.bounds_history)

class BackgroundAnalysis:
    """ Stores background analysis statistics. """
    def __init__(self):

        self.threshold = None
        self.average_delta = None
        self.max_temp = None
        self.min_temp = None
        self.mean_temp = None

class FrameBuffer:
    """ Stores entire clip in memory, required for some operations such as track exporting. """
    def __init__(self):
        self.thermal = None
        self.filtered = None
        self.delta = None
        self.mask = None
        self.flow = None
        self.reset()

    @property
    def has_flow(self):
        return self.flow and len(self.flow) != 0

    def generate_flow(self, opt_flow):
        """
        Generate optical flow from thermal frames
        :param opt_flow: An optical flow algorithm
        """

        self.flow = []

        height, width = self.filtered[0].shape
        flow = np.zeros([height, width, 2], dtype=np.uint8)

        current = None
        for next in self.filtered:
            if current is not None:
                current_gray_frame = (current / 2).astype(np.uint8)
                next_gray_frame = (next / 2).astype(np.uint8)
                flow = opt_flow.calc(current_gray_frame, next_gray_frame, flow)

            current = next

            self.flow.append(flow.copy())

    def reset(self):
        """
        Empties buffer
        """
        self.thermal = []
        self.filtered = []
        self.delta = []
        self.mask = []
        self.flow = []

    def __len__(self):
        return len(self.thermal)

class TrackExtractor:
    """ Extracts tracks from a stream of frames. """

    # auto threshold needs to find a near maximum value to calculate the threshold level
    # a better solution might be the mean of the max of each frame?
    THRESHOLD_PERCENTILE = 99.9

    # the dimensions of the tracks in pixels (width and height)
    WINDOW_SIZE = 64

    # if the mean pixel change is below this threshold then classify the video as having a static background
    STATIC_BACKGROUND_THRESHOLD = 5.0

    # any clips with a mean temperature hotter than this will be excluded
    MAX_MEAN_TEMPERATURE_THRESHOLD = 3800

    # any clips with a temperature dynamic range greater than this will be excluded
    MAX_TEMPERATURE_RANGE_THRESHOLD = 2000

    # number of pixels around object to pad.
    FRAME_PADDING = 8

    # number of frames to wait before deleting a lost track
    DELETE_LOST_TRACK_FRAMES = 9

    def __init__(self):

        # start time of video
        self.video_start_time = None
        # name of source file
        self.source_file = None
        # cptv reader
        self.reader = None
        # dictionary containing various statistics about the clip / tracking process.
        self.stats = {}
        # when enabled uses high quality optical flow, which is much slower.  Usualy the default settings are fine.
        self.high_quality_optical_flow = False
        # used to calculate optical flow
        self.opt_flow = None
        # the background to use during background subtraction
        self.background = None
        # how much hotter target must be from background to trigger a region of interest.
        self.threshold = None
        # the current frame number
        self.frame_on = None
        # enables verbose mode
        self.verbose = False
        # maximum number of tracks to extract from a clip.  Takes the n best tracks.  Set to None for unlimited.
        self.max_tracks = 10

        # filters for tracks
        self.track_min_duration = 3.0
        self.track_min_offset = 4.0
        self.track_min_delta = 1.0
        self.track_min_mass = 2.0

        # a list of currently active tracks
        self.active_tracks = []
        # a list of all tracks
        self.tracks = []
        # list of regions for each frame
        self.region_history = []

        # this buffers store the entire video in memory and are required for fast track exporting
        self.frame_buffer = FrameBuffer()

        # the previous filtered frame
        self._prev_filtered = None

    def load(self, filename):
        """
        Loads a cptv file, and prepares for track extraction.
        """
        self.source_file = filename
        self.reader = CPTVReader(open(filename, 'rb'))
        self.video_start_time = self.reader.timestamp
        self.stats.update(self.get_video_stats())

    def extract_tracks(self):
        """
        Extracts tracks from given source.  Setting self.tracks to a list of good tracks with the clip
        :param source_file: filename of cptv file to process
        """

        assert self.reader, "Must call load before extracting tracks."

        # we need to load the entire video so we can analyse the background.
        frames = [frame for frame, offset in self.reader]

        # first we get the background.  This requires reading the entire source into memory.
        self.background, background_stats= self.analyse_background(frames)

        self.stats['threshold'] = background_stats.threshold
        self.stats['average_background_delta'] = background_stats.average_delta
        self.stats['mean_temp'] = background_stats.mean_temp
        self.stats['max_temp'] = background_stats.max_temp
        self.stats['min_temp'] = background_stats.min_temp

        self.threshold = background_stats.threshold

        # exclude clips with moving backgrounds
        if background_stats.average_delta > self.STATIC_BACKGROUND_THRESHOLD:
            return

        # don't process clips that are too hot.
        if background_stats.mean_temp > self.MAX_MEAN_TEMPERATURE_THRESHOLD:
            return

        # don't process clips with too large of a temperature difference
        if background_stats.max_temp - background_stats.min_temp > self.MAX_TEMPERATURE_RANGE_THRESHOLD:
            return

        # reset the track ID so we start at 1
        Track._track_id = 1
        self.tracks = []
        self.active_tracks = []
        self.region_history = []

        # create optical flow
        self.opt_flow = cv2.createOptFlow_DualTVL1()
        if not self.high_quality_optical_flow:
            # see https://stackoverflow.com/questions/19309567/speeding-up-optical-flow-createoptflow-dualtvl1
            self.opt_flow.setTau(1 / 4)
            self.opt_flow.setScalesNumber(3)
            self.opt_flow.setWarpingsNumber(3)
            self.opt_flow.setScaleStep(0.5)

        # process each frame
        self.frame_on = 0
        for frame in frames:
            self.track_next_frame(frame)

        # filter out tracks that do not move, or look like noise
        self.filter_tracks()

    def track_next_frame(self, frame):
        """
        Tracks objects through frame
        :param frame: A numpy array of shape (height, width) and type uint16
        """

        assert self.opt_flow is not None, "Optical flow not initialised."
        assert self.background is not None, "Background not initialised."

        filtered = self._get_filtered(frame)
        regions, mask = self.get_regions(filtered, self._prev_filtered)

        # save history
        self.frame_buffer.thermal.append(frame)
        self.frame_buffer.filtered.append(filtered)
        self.frame_buffer.mask.append(mask)

        self.region_history.append(regions)

        self.apply_matchings(regions)
        self._prev_filtered = filtered.copy()
        self.frame_on += 1

    def export_tracks(self, database: TrackDatabase):
        """
        Writes tracks to a track database.
        :param database: database to write track to.
        """

        clip_id = os.path.basename(self.source_file)

        # overwrite any old clips.
        # Note: we do this even if there are no tracks so there there will be a blank clip entry as a record
        # that we have processed it.
        database.create_clip(clip_id)

        if len(self.tracks) == 0:
            return

        if not self.frame_buffer.has_flow:
            self.frame_buffer.generate_flow(self.opt_flow)

        # get track data
        for track_number, track in enumerate(self.tracks):
            track_data = []
            for i in range(len(track)):
                channels = self.get_track_channels(track, i)
                track_data.append(channels)
            track_data = np.int16(track_data)
            track_id = track_number+1
            database.add_track(clip_id, track_id, track_data)

            # todo: save stats
            # tracker.save_stats(stats_path_and_filename)

    def get_track_channels(self, track: Track, frame_number):
        """
        Gets frame channels for track at given frame number.  If frame number outside of track's lifespan an exception
        is thrown.  Requires the frame_buffer to be filled.
        :param track: the track to get frames for.
        :param frame_number: the frame number where 0 is the first frame of the track.
        :return: numpy array of size [64,64,5] where channels are thermal, filtered, u, v, mask
        """

        if frame_number < 0 or frame_number >= len(track):
            raise Exception("Frame {} is out of bounds for track with {} frames".format(
                frame_number, len(track))
            )

        bounds = track.bounds_history[frame_number]
        tracker_frame = track.start_frame + frame_number

        # window size must be even for get_image_subsection to work.
        window_size = (max(self.WINDOW_SIZE, bounds.width, bounds.height) // 2) * 2

        if tracker_frame < 0 or tracker_frame >= len(self.frame_buffer.thermal):

            print(len(self.frame_buffer.thermal))
            print(len(track))
            print(track.start_frame)
            print(len(self.frame_buffer.filtered))

            raise Exception("Track frame is out of bounds.  Frame {} was expected to be between [0-{}]".format(
               tracker_frame, len(self.frame_buffer.thermal)-1))

        thermal = get_image_subsection(self.frame_buffer.thermal[tracker_frame], bounds, (window_size, window_size))
        filtered = get_image_subsection(self.frame_buffer.filtered[tracker_frame], bounds,
                                        (window_size, window_size), 0)
        flow = get_image_subsection(self.frame_buffer.flow[tracker_frame], bounds, (window_size, window_size), 0)
        mask = get_image_subsection(self.frame_buffer.mask[tracker_frame], bounds, (window_size, window_size), 0)

        if window_size != self.WINDOW_SIZE:
            scale = self.WINDOW_SIZE / window_size
            thermal = scipy.ndimage.zoom(np.float32(thermal), (scale, scale), order=1)
            filtered = scipy.ndimage.zoom(np.float32(filtered), (scale, scale), order=1)
            flow = scipy.ndimage.zoom(np.float32(flow), (scale, scale, 1), order=1)
            mask = scipy.ndimage.zoom(np.float32(mask), (scale, scale), order=1)

        # make sure only our pixels are included in the mask.
        mask[mask != bounds.id] = 0
        mask[mask > 0] = 1

        # stack together into a numpy array.
        # by using int16 we loose a little precision on the filtered frames, but not much (only 1 unit)
        frame = np.int16(np.stack((thermal, filtered, flow[:, :, 0], flow[:, :, 1], mask), axis=2))

        return frame

    def apply_matchings(self, regions):
        """
        Work out the best matchings between tracks and regions of interest for the current frame.
        Create any new tracks required.
        """
        scores = []
        for track in self.active_tracks:
            for region in regions:
                distance, size_change = track.get_track_region_score(region)

                # we give larger tracks more freedom to find a match as they might move quite a bit.
                max_distance = np.clip(7 * (track.mass ** 0.5), 30, 95)
                size_change = np.clip(track.mass, 50, 500)

                if distance > max_distance:
                    continue
                if size_change > size_change:
                    continue
                scores.append((distance, track, region))

        # apply matchings greedily.  Low score is best.
        matched_tracks = set()
        used_regions = set()

        scores.sort(key=lambda record: record[0])
        results = []

        for (score, track, region) in scores:
            # don't match a track twice
            if track in matched_tracks or region in used_regions:
                continue
            track.add_frame(region)
            used_regions.add(region)
            matched_tracks.add(track)
            results.append((track, score))

        # create new tracks for any unmatched regions
        for region in regions:
            if region in used_regions:
                continue
            # make sure we don't overlap with existing tracks.  This can happen if a tail gets tracked as a new object
            overlaps = [track.bounds.overlap_area(region) for track in self.active_tracks]
            if len(overlaps) > 0 and max(overlaps) > (region.area * 0.25):
                continue
            track = Track()
            track.add_frame(region)
            track.start_time = self.video_start_time + datetime.timedelta(seconds=self.frame_on / 9.0)
            track.start_frame = self.frame_on
            track.tracker = self
            self.active_tracks.append(track)
            self.tracks.append(track)

        # check if any tracks did not find a matched region
        for track in [track for track in self.active_tracks if track not in matched_tracks]:
            # we lost this track.  start a count down, and if we don't get it back soon remove it
            track.frames_since_target_seen += 1

        # remove any tracks that have not seen their target in 9 frames
        self.active_tracks = [track for track in self.active_tracks if track.frames_since_target_seen < self.DELETE_LOST_TRACK_FRAMES]

    def filter_tracks(self):

        track_stats = [(track.get_stats(), track) for track in self.tracks]
        track_stats.sort(reverse=True, key=lambda record : record[0].score)

        if self.verbose:
            for stats, track in track_stats:
                print(" - track duration:{:.1f}sec offset:{:.1f}px delta:{:.1f} mass:{:.1f}px".format(
                    stats.duration, stats.max_offset, stats.delta_std, stats.average_mass
                ))

        # filter out tracks that probably are just noise.
        good_tracks = []
        for stats, track in track_stats:

            # discard any tracks that are less than 3 seconds long (27 frames)
            # these are probably glitches anyway, or don't contain enough information.
            if stats.duration < self.track_min_duration:
                continue

            # discard tracks that do not move enough
            if stats.max_offset < self.track_min_offset:
                continue

            # discard tracks that do not have enough delta within the window (i.e. pixels that change a lot)
            if stats.delta_std < self.track_min_delta:
                continue

            # discard tracks that do not have enough enough average mass.
            if stats.average_mass < self.track_min_mass:
                continue

            good_tracks.append(track)

        self.tracks = good_tracks

        # apply max_tracks filter
        # note, we take the n best tracks.
        if self.max_tracks is not None and self.max_tracks < len(self.tracks):
            print(" -using only {0} tracks out of {1}".format(self.max_tracks, len(self.tracks)))
            self.tracks = self.tracks[:self.max_tracks]

    def _get_filtered(self, thermal):
        """
        Calculates the background removed, filtered frame.
        :param thermal: source thermal frame
        :return: a filtered frame
        """
        filtered = thermal - self.background
        filtered = filtered - np.median(filtered)
        filtered[filtered < 0] = 0
        return filtered

    def get_regions(self, filtered, prev_filtered=None):
        """
        Calculates pixels of interest mask from filtered image, and returns both the labeled mask and their bounding
        rectangles.
        :param filtered: The filtered frame
        :param prev_filtered: The previous filtered frame, required for pixel deltas to be calculated
        :return: regions of interest, mask frame
        """

        thresh = np.asarray((apply_threshold(filtered, threshold=self.threshold)), dtype=np.uint8)

        # perform erosion
        # this removes small slivers
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=1)
        labels, mask, stats, centroids = cv2.connectedComponentsWithStats(eroded)

        # we enlarge the rects a bit, partly because we eroded them previously, and partly because we want some context.
        padding = self.FRAME_PADDING

        # get frames change
        if prev_filtered is not None:
            # we need a lot of percision because the values are squared.  Float32 tends to get set to infinity.
            delta_frame = np.abs(np.float16(filtered) - np.float16(prev_filtered), dtype=np.float64)
        else:
            delta_frame = None

        # find regions of interest
        regions = []
        for i in range(1, labels):
            region = Region(
                stats[i, 0] - padding, stats[i, 1] - padding, stats[i, 2] + padding * 2,
                stats[i, 3] + padding * 2, stats[i, 4], i
            )
            if delta_frame is not None:
                region_difference = get_image_subsection(delta_frame, region, (self.WINDOW_SIZE, self.WINDOW_SIZE), 0)
                region.pixel_variance = np.var(region_difference)
            regions.append(region)

        return regions , mask

    def get_video_stats(self):
        """
        Extracts useful statics from video clip.
        :param source:
        :returns: a dictionary containing the video statistics.
        """
        local_tz = pytz.timezone('Pacific/Auckland')
        result = {}
        result['date_time'] = self.video_start_time.astimezone(local_tz)
        result['is_night'] = self.video_start_time.astimezone(
            local_tz).time().hour >= 21 or self.video_start_time.astimezone(local_tz).time().hour <= 4

        return result

    def analyse_background(self, frames):
        """
        Runs through all provided frames and estimates the background, consuming all the source frames.
        :param frames: a list of numpy array frames
        :return: background, background_stats
        """

        # note: unfortunately this must be done before any other processing, which breaks the streaming architecture
        # for this reason we must return all the frames so they can be reused
        frames = np.asarray(frames, dtype=np.float32)

        background = np.percentile(np.asarray(frames), q=10.0, axis=0)
        filtered = np.reshape(frames - background, [-1])

        delta = np.asarray(frames[1:], dtype=np.float32) - np.asarray(frames[:-1], dtype=np.float32)
        average_delta = float(np.mean(np.abs(delta)))

        threshold = float(np.percentile(filtered, q=TrackExtractor.THRESHOLD_PERCENTILE) / 2)

        # cap the threshold to something reasonable
        if threshold < 10.0:
            threshold = 10.0
        if threshold > 50.0:
            threshold = 50.0

        background_stats = BackgroundAnalysis()
        background_stats.threshold = float(threshold)
        background_stats.average_delta = float(average_delta)
        background_stats.min_temp = float(np.min(frames))
        background_stats.max_temp = float(np.max(frames))
        background_stats.mean_temp = float(np.mean(frames))

        return background, background_stats


def apply_threshold(frame, threshold):
    """
    Creates a binary mask out of an image by applying a threshold.
    Any pixels more than the threshold are set 1, all others are set to 0.
    A blur is also applied as a filtering step
    """
    thresh = cv2.GaussianBlur(frame.astype(np.float32), (5, 5), 0) - threshold
    thresh[thresh < 0] = 0
    thresh[thresh > 0] = 1
    return thresh