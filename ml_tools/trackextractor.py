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
import pytz

import numpy as np
import cv2
import h5py
import scipy.ndimage
import logging

from ml_tools.tools import get_image_subsection, blosc_zstd
from ml_tools.tools import Rectangle
from ml_tools.trackdatabase import TrackDatabase
from ml_tools import tools
from ml_tools.dataset import TrackChannels

from cptv import CPTVReader

__import__('tables')  # <-- import PyTables; __import__ so that linters don't complain


class Region(Rectangle):
    """ Region is a rectangle extended to support mass. """
    def __init__(self, topleft_x, topleft_y, width, height, mass=0, pixel_variance=0, id=0, frame_index=0, was_cropped=False):
        super().__init__(topleft_x, topleft_y, width, height)
        # number of active pixels in region
        self.mass = mass
        # how much pixels in this region have changed since last frame
        self.pixel_variance = pixel_variance
        # an identifier for this region
        self.id = id
        # frame index from clip
        self.frame_index = frame_index
        # if this region was cropped or not
        self.was_cropped = was_cropped

    def copy(self):
        return Region(
            self.x, self.y, self.width, self.height, self.mass, self.pixel_variance, self.id, self.frame_index,
            self.was_cropped
        )

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

        # the tag for this track
        self.tag = "unknown"

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

    def add_blank_frame(self):
        """ Maintains same bounds as previously, does not reset framce_since_target_seen counter """
        self.bounds_history.append(self.bounds.copy())
        self.bounds.mass = 0
        self.bounds.pixel_variance = 0
        self.bounds.frame_index += 1
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
        score = min(movement_points,100) + min(delta_points, 100)

        stats = TrackMovementStatistics(
            movement=float(movement),
            max_offset=float(max_offset),
            average_mass=float(np.mean(mass_history)),
            median_mass=float(np.median(mass_history)),
            duration=len(self) / 9.0,
            delta_std=float(delta_std),
            score=float(score)
        )

        return stats

    def smooth(self, frame_bounds: Rectangle):
        """
        Smooths out any quick changes in track dimensions
        :param frame_bounds The boundaries of the video frame.
        """
        if len(self.bounds_history) == 0:
            return

        new_bounds_history = []
        prev_frame = self.bounds_history[0]
        current_frame = self.bounds_history[0]
        next_frame = self.bounds_history[1]

        for i in range(len(self.bounds_history)):

            prev_frame = self.bounds_history[max(0, i-1)]
            current_frame = self.bounds_history[i]
            next_frame = self.bounds_history[min(len(self.bounds_history)-1, i+1)]

            frame_x = current_frame.mid_x
            frame_y = current_frame.mid_y
            frame_width = (prev_frame.width + current_frame.width + next_frame.width) / 3
            frame_height = (prev_frame.height + current_frame.height + next_frame.height) / 3
            frame = Region(int(frame_x - frame_width / 2), int(frame_y - frame_height / 2), int(frame_width), int(frame_height))
            frame.crop(frame_bounds)

            new_bounds_history.append(frame)

        self.bounds_history = new_bounds_history

    def trim(self):
        """
        Removes empty frames from start and end of track
        """
        mass_history = [int(bound.mass) for bound in self.bounds_history]

        start = 0
        while start < len(self) and mass_history[start] <= 2:
            start += 1
        end = len(self)-1
        while end > 0 and mass_history[end] <= 2:
            end -= 1

        if end < start:
            end = start

        self.start_time += datetime.timedelta(seconds=start / 9.0)
        self.start_frame += start
        self.bounds_history = self.bounds_history[start:end-1]

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

    def get_overlap_ratio(self, other_track, threshold = 0.05):
        """
        Checks what ratio of the time these two tracks overlap.
        :param other_track: the other track to compare with
        :param threshold: how much frames must be overlapping to be counted
        :return: the ratio of frames that overlap
        """

        if len(self) == 0 or len(other_track) == 0:
            return 0.0

        start = max(self.start_frame, other_track.start_frame)
        end = min(self.start_frame + len(self), other_track.start_frame + len(other_track))

        frames_overlapped = 0

        for pos in range(start, end+1):
            our_index = pos - self.start_frame
            other_index = pos - other_track.start_frame
            if our_index >= 0 and other_index >= 0 and our_index < len(self) and other_index < len(other_track):
                our_bounds = self.bounds_history[our_index]
                other_bounds = other_track.bounds_history[other_index]
                overlap = our_bounds.overlap_area(other_bounds) / our_bounds.area
                if overlap >= threshold:
                    frames_overlapped += 1

        return frames_overlapped / len(self)

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

    @property
    def end_time(self):
        return self.start_time + datetime.timedelta(seconds=self.duration)

    def __repr__(self):
        return "Track:{} frames".format(len(self))

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
        self.background_deviation = None

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
        return self.flow is not None and len(self.flow) != 0

    def generate_flow(self, opt_flow, flow_threshold=40):
        """
        Generate optical flow from thermal frames
        :param opt_flow: An optical flow algorithm
        """

        self.flow = []

        height, width = self.filtered[0].shape
        flow = np.zeros([height, width, 2], dtype=np.float32)

        current = None
        for frame in self.thermal:
            frame = np.float32(frame)
            # strong filtering helps with the optical flow.
            threshold = np.median(frame) + flow_threshold
            next = np.uint8(np.clip(frame - threshold, 0, 255))

            if current is not None:
                # for some reason openCV spins up lots of threads for this which really slows things down, so we
                # cap the threads to 2
                cv2.setNumThreads(2)
                flow = opt_flow.calc(current, next, flow)

            current = next

            # scale up the motion vectors so that we get some additional precision
            # but also make sure they fit within an int16
            scaled_flow = np.clip(flow * 256, -16000, 16000)
            self.flow.append(scaled_flow)

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

    # enables fast BLOSC compression (requires plugin)
    ENABLE_COMPRESSION = False

    # includes the filtered channel when exporting.  This is typically not used.  If compression is enabled the
    # filesize can be reduced by not including it.
    INCLUDE_FILTERED_CHANNEL = True

    # auto threshold needs to find a near maximum value to calculate the threshold level
    # a better solution might be the mean of the max of each frame?
    THRESHOLD_PERCENTILE = 99.9

    # measures how different pixels are from estimated background, if they are on average less different than this then
    # video is considered to be static.
    STATIC_BACKGROUND_THRESHOLD = 4.0

    # any clips with a mean temperature hotter than this will be excluded
    MAX_MEAN_TEMPERATURE_THRESHOLD = 10000

    # any clips with a temperature dynamic range greater than this will be excluded
    MAX_TEMPERATURE_RANGE_THRESHOLD = 10000

    # number of pixels around object to pad.
    FRAME_PADDING = 6

    # number of frames to wait before deleting a lost track
    DELETE_LOST_TRACK_FRAMES = 9

    # strategy to use when dealing with regions of interest that are cropped against the side of the frame
    # in general these regions often do not have enough information to accurately identify the animal.
    # options are
    # 'all': All cropped regions are included, good for classifier
    # 'cautious': Regions that are only cropped a bit are let through, this is good for training data
    # 'none': No cropped regions are permitted.  This is the most safe.
    CROPPED_REGIONS_STRATEGY = "cautious"

    # when enabled smooths tracks so that track dimensions do not change too quickly.
    TRACK_SMOOTHING = False

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
        # how much hotter target must be from background to trigger a region of interest.
        self.threshold = 0
        # the current frame number
        self.frame_on = 0
        # enables verbose mode
        self.verbose = False
        # maximum number of tracks to extract from a clip.  Takes the n best tracks.  Set to None for unlimited.
        self.max_tracks = 10

        # per frame temperature statistics for thermal channel
        self.frame_stats_min = []
        self.frame_stats_max = []
        self.frame_stats_median = []
        self.frame_stats_mean = []

        # filters for tracks
        self.track_min_duration = 3.0
        self.track_min_offset = 4.0
        self.track_min_delta = 1.0
        self.track_min_mass = 2.0

        # minimum allowed threshold for mask, smaller values detect more objects, but bring up additional false positives
        self.min_threshold = 30

        # how much to threshold thermal before calculating optical flow.
        self.flow_threshold = 40

        # reason qwhy clip was rejected, or none if clip was accepted
        self.reject_reason = None

        # a list of currently active tracks
        self.active_tracks = []
        # a list of all tracks
        self.tracks = []
        # list of regions for each frame
        self.region_history = []

        # if enabled will force the background subtraction algorithm off.
        self.disable_background_subtraction = False
        # rejects any videos that have non static backgrounds
        self.reject_non_static_clips = False

        # this buffers store the entire video in memory and are required for fast track exporting
        self.frame_buffer = FrameBuffer()

        # the previous filtered frame
        self._prev_filtered = None

        # accumulates frame changes for FM_DELTA algorithm
        self.accumulator = None


    def load(self, filename):
        """
        Loads a cptv file, and prepares for track extraction.
        """
        self.source_file = filename
        self.reader = CPTVReader(open(filename, 'rb'))
        local_tz = pytz.timezone('Pacific/Auckland')
        self.video_start_time = self.reader.timestamp.astimezone(local_tz)
        self.stats.update(self.get_video_stats())

    def extract_tracks(self):
        """
        Extracts tracks from given source.  Setting self.tracks to a list of good tracks within the clip
        :param source_file: filename of cptv file to process
        :returns: True if clip was successfully processed, false otherwise
        """

        assert self.reader, "Must call load before extracting tracks."

        self.reject_reason = None

        # we need to load the entire video so we can analyse the background.
        frames = [frame for frame, offset in self.reader]
        self.frame_buffer.thermal = frames

        # first we get the background.  This requires reading the entire source into memory.
        background, background_stats = self.analyse_background(frames)
        is_static_background = background_stats.background_deviation < self.STATIC_BACKGROUND_THRESHOLD

        self.stats['threshold'] = background_stats.threshold
        self.stats['average_background_delta'] = background_stats.background_deviation
        self.stats['average_delta'] = background_stats.average_delta
        self.stats['mean_temp'] = background_stats.mean_temp
        self.stats['max_temp'] = background_stats.max_temp
        self.stats['min_temp'] = background_stats.min_temp
        self.stats['is_static'] = is_static_background

        self.threshold = background_stats.threshold

        # if the clip is moving then remove the estimated background and just use a threshold.
        if not is_static_background or self.disable_background_subtraction:
            background = None

        if len(frames) <= 9:
            self.reject_reason = "Clip too short {} frames".format(len(frames))
            return False

        if self.reject_non_static_clips and not is_static_background:
            self.reject_reason = "Non static background deviation={:.1f}".format(background_stats.background_deviation)
            return False

        # don't process clips that are too hot.
        if self.MAX_MEAN_TEMPERATURE_THRESHOLD and background_stats.mean_temp > self.MAX_MEAN_TEMPERATURE_THRESHOLD:
            self.reject_reason = "Mean temp too high {}".format(background_stats.mean_temp)
            return False

        # don't process clips with too large of a temperature difference
        if self.MAX_TEMPERATURE_RANGE_THRESHOLD and (background_stats.max_temp - background_stats.min_temp > self.MAX_TEMPERATURE_RANGE_THRESHOLD):
            self.reject_reason = "Temp delta too high {}".format(background_stats.max_temp - background_stats.min_temp)
            return False

        # reset the track ID so we start at 1
        Track._track_id = 1
        self.tracks = []
        self.active_tracks = []
        self.region_history = []

        # create optical flow
        self.opt_flow = cv2.createOptFlow_DualTVL1()
        self.opt_flow.setUseInitialFlow(True)
        if not self.high_quality_optical_flow:
            # see https://stackoverflow.com/questions/19309567/speeding-up-optical-flow-createoptflow-dualtvl1
            self.opt_flow.setTau(1 / 4)
            self.opt_flow.setScalesNumber(3)
            self.opt_flow.setWarpingsNumber(3)
            self.opt_flow.setScaleStep(0.5)

        # process each frame
        self.frame_on = 0
        for frame in frames:
            self.track_next_frame(frame, background)

        # filter out tracks that do not move, or look like noise
        self.filter_tracks()

        # apply smoothing if required
        if self.TRACK_SMOOTHING and len(frames) > 0:
            frame_height, frame_width = frames[0].shape
            for track in self.tracks:
                track.smooth(Rectangle(0,0,frame_width, frame_height))

        return True

    def get_filtered(self, thermal, background=None):
        """
        Calculates filtered frame from thermal
        :param thermal: the thermal frame
        :param background: (optional) used for background subtraction
        :return: the filtered frame
        """

        thermal = np.float32(thermal)

        if background is None:
            filtered = thermal - np.median(thermal) - 40
            filtered[filtered < 0] = 0
        else:
            background = np.float32(background)
            filtered = thermal - background
            filtered[filtered < 0] = 0
            filtered = filtered - np.median(filtered)
            filtered[filtered < 0] = 0

        return filtered

    def track_next_frame(self, thermal, background=None):
        """
        Tracks objects through frame
        :param thermal: A numpy array of shape (height, width) and type uint16
        :param background: (optional) Background image, a numpy array of shape (height, width) and type uint16
            If specified background subtraction algorithm will be used.
        """

        thermal = np.float32(thermal)
        filtered = self.get_filtered(thermal, background)

        regions, mask = self.get_regions_of_interest(filtered, self._prev_filtered)

        # save frame stats
        self.frame_stats_min.append(np.min(thermal))
        self.frame_stats_max.append(np.max(thermal))
        self.frame_stats_median.append(np.median(thermal))
        self.frame_stats_mean.append(np.mean(thermal))

        # save history
        self.frame_buffer.filtered.append(np.float32(filtered))
        self.frame_buffer.mask.append(np.float32(mask))

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
        database.create_clip(clip_id, self)

        if len(self.tracks) == 0:
            return

        if not self.frame_buffer.has_flow:
            self.frame_buffer.generate_flow(self.opt_flow, self.flow_threshold)

        # get track data
        for track_number, track in enumerate(self.tracks):
            track_data = []
            for i in range(len(track)):
                channels = self.get_track_channels(track, i)

                # zero out the filtered channel
                if not self.INCLUDE_FILTERED_CHANNEL:
                    channels[TrackChannels.filtered] = 0
                track_data.append(channels)
            track_id = track_number+1
            database.add_track(clip_id, track_id, track_data, track, opts=blosc_zstd if self.ENABLE_COMPRESSION else None)

    def get_track_channels(self, track: Track, frame_number):
        """
        Gets frame channels for track at given frame number.  If frame number outside of track's lifespan an exception
        is thrown.  Requires the frame_buffer to be filled.
        :param track: the track to get frames for.
        :param frame_number: the frame number where 0 is the first frame of the track.
        :return: numpy array of size [channels, height, width] where channels are thermal, filtered, u, v, mask
        """

        if frame_number < 0 or frame_number >= len(track):
            raise ValueError("Frame {} is out of bounds for track with {} frames".format(
                frame_number, len(track))
            )

        bounds = track.bounds_history[frame_number]
        tracker_frame = track.start_frame + frame_number

        if tracker_frame < 0 or tracker_frame >= len(self.frame_buffer.thermal):
            raise ValueError("Track frame is out of bounds.  Frame {} was expected to be between [0-{}]".format(
               tracker_frame, len(self.frame_buffer.thermal)-1))

        thermal = get_image_subsection(self.frame_buffer.thermal[tracker_frame], bounds)
        filtered = get_image_subsection(self.frame_buffer.filtered[tracker_frame], bounds)
        flow = get_image_subsection(self.frame_buffer.flow[tracker_frame], bounds)
        mask = get_image_subsection(self.frame_buffer.mask[tracker_frame], bounds)

        # make sure only our pixels are included in the mask.
        mask[mask != bounds.id] = 0
        mask[mask > 0] = 1

        # stack together into a numpy array.
        # by using int16 we loose a little precision on the filtered frames, but not much (only 1 bit)
        frame = np.int16(np.stack((thermal, filtered, flow[:, :, 0], flow[:, :, 1], mask), axis=0))

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
        new_tracks = set()

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
            new_tracks.add(track)
            self.active_tracks.append(track)
            self.tracks.append(track)

        # check if any tracks did not find a matched region
        for track in [track for track in self.active_tracks if track not in matched_tracks and track not in new_tracks]:
            # we lost this track.  start a count down, and if we don't get it back soon remove it
            track.frames_since_target_seen += 1
            track.add_blank_frame()

        # remove any tracks that have not seen their target in 9 frames
        self.active_tracks = [track for track in self.active_tracks if track.frames_since_target_seen < self.DELETE_LOST_TRACK_FRAMES]

    def filter_tracks(self):

        for track in self.tracks:
            track.trim()

        track_stats = [(track.get_stats(), track) for track in self.tracks]
        track_stats.sort(reverse=True, key=lambda record : record[0].score)

        if self.verbose:
            for stats, track in track_stats:
                print(" - track duration:{:.1f}sec offset:{:.1f}px delta:{:.1f} mass:{:.1f}px".format(
                    stats.duration, stats.max_offset, stats.delta_std, stats.average_mass
                ))

        # find how much each track overlaps with other tracks

        track_overlap_ratio = {}

        for track in self.tracks:
            highest_ratio = 0
            for other in self.tracks:
                if track == other:
                    continue
                highest_ratio = max(track.get_overlap_ratio(other), highest_ratio)
            track_overlap_ratio[track] = highest_ratio

        # filter out tracks that probably are just noise.
        good_tracks = []
        for stats, track in track_stats:

            # discard any tracks that overlap too often with other tracks.  This normally means we are tracking the
            # tail of an animal.
            if track_overlap_ratio[track] > 0.5:
                continue

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
            logging.warning(" -using only {0} tracks out of {1}".format(self.max_tracks, len(self.tracks)))
            self.tracks = self.tracks[:self.max_tracks]

    def get_regions_of_interest(self, filtered, prev_filtered=None):
        """
        Calculates pixels of interest mask from filtered image, and returns both the labeled mask and their bounding
        rectangles.
        :param filtered: The filtered frame
        :param prev_filtered: The previous filtered frame, required for pixel deltas to be calculated
        :return: regions of interest, mask frame
        """

        frame_height, frame_width = filtered.shape

        # get frames change
        if prev_filtered is not None:
            # we need a lot of precision because the values are squared.  Float32 should work.
            delta_frame = np.abs(np.float32(filtered) - np.float32(prev_filtered))
        else:
            delta_frame = None

        thresh = np.uint8(apply_threshold(filtered, threshold=self.threshold))

        # applies erosion
        erosion = 0

        if erosion:
            # this removes small slivers
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations=erosion)

        labels, mask, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        # we enlarge the rects a bit, partly because we eroded them previously, and partly because we want some context.
        padding = self.FRAME_PADDING

        # find regions of interest
        regions = []
        for i in range(1, labels):

            region = Region(
                stats[i, 0] - padding,
                stats[i, 1] - padding,
                stats[i, 2] + padding * 2,
                stats[i, 3] + padding * 2,
                stats[i, 4], 0, i, self.frame_on
            )

            old_region = region.copy()
            if self.CROPPED_REGIONS_STRATEGY == "all":
                # in this case we just clip the region to the edges of the screen
                region.left = max(region.left, 0)
                region.top = max(region.top, 0)
                region.right = min(region.right, frame_width - 1)
                region.bottom = min(region.bottom, frame_height - 1)
            elif self.CROPPED_REGIONS_STRATEGY == "cautious":
                # keep cropped regions if they have a reasonable size
                region.left = max(region.left, 0)
                region.top = max(region.top, 0)
                region.right = min(region.right, frame_width - 1)
                region.bottom = min(region.bottom, frame_height - 1)

                crop_width_fraction = (old_region.width - region.width) / old_region.width
                crop_height_fraction = (old_region.height - region.height) / old_region.height

                if crop_width_fraction > 0.25 or crop_height_fraction > 0.25:
                    continue

            elif self.CROPPED_REGIONS_STRATEGY == "none":
                # all regions that touch the side of the screen are removed
                if (region.left < 0) or (region.right > frame_width) or (region.top < 0) or (region.top < 0) or (region.bottom > frame_height):
                    continue
            else:
                raise ValueError(
                    "Invalid mode for CROPPED_REGIONS_STRATEGY, expected ['all','cautious','none'] but found {}".format(
                        self.CROPPED_REGIONS_STRATEGY))
            region.was_cropped = str(old_region) != str(region)

            if delta_frame is not None:
                region_difference = np.float32(get_image_subsection(delta_frame, region))
                region.pixel_variance = np.var(region_difference)

            # filter out regions that are probably just noise
            if region.pixel_variance < 2.0 and region.mass < 8:
                continue

            regions.append(region)

        return regions , mask

    def get_video_stats(self):
        """
        Extracts useful statics from video clip.
        :returns: a dictionary containing the video statistics.
        """
        local_tz = pytz.timezone('Pacific/Auckland')
        result = {}
        result['date_time'] = self.video_start_time.astimezone(local_tz)
        result['is_night'] = self.video_start_time.astimezone(
            local_tz).time().hour >= 21 or self.video_start_time.astimezone(local_tz).time().hour <= 4

        return result

    def analyse_background(self, frames_list):
        """
        Runs through all provided frames and estimates the background, consuming all the source frames.
        :param frames_list: a list of numpy array frames
        :return: background, background_stats
        """

        # note: unfortunately this must be done before any other processing, which breaks the streaming architecture
        # for this reason we must return all the frames so they can be reused

        frames = np.float32(frames_list)
        background = np.percentile(frames, q=10, axis=0)
        filtered = np.float32([self.get_filtered(frame, background) for frame in frames_list])

        delta = np.asarray(frames[1:], dtype=np.float32) - np.asarray(frames[:-1], dtype=np.float32)
        average_delta = float(np.mean(np.abs(delta)))

        # take half the max filtered value as a threshold
        threshold = float(np.percentile(np.reshape(filtered, [-1]), q=TrackExtractor.THRESHOLD_PERCENTILE) / 2)

        # cap the threshold to something reasonable
        if threshold < self.min_threshold:
            threshold = self.min_threshold
        if threshold > 50.0:
            threshold = 50.0

        background_stats = BackgroundAnalysis()
        background_stats.threshold = float(threshold)
        background_stats.average_delta = float(average_delta)
        background_stats.min_temp = float(np.min(frames))
        background_stats.max_temp = float(np.max(frames))
        background_stats.mean_temp = float(np.mean(frames))
        background_stats.background_deviation = float(np.mean(np.abs(filtered)))

        return background, background_stats


def apply_threshold(frame, threshold):
    """
    Creates a binary mask out of an image by applying a threshold.
    Any pixels more than the threshold are set 1, all others are set to 0.
    A blur is also applied as a filtering step
    """
    thresh = cv2.GaussianBlur(np.float32(frame), (5, 5), 0) - threshold
    thresh[thresh < 0] = 0
    thresh[thresh > 0] = 1
    return thresh