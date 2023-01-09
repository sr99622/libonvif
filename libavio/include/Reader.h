/********************************************************************
* libavio/include/Reader.h
*
* Copyright (c) 2022  Stephen Rhodes
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*********************************************************************/

#ifndef READER_H
#define READER_H

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "Exception.h"
#include "Queue.h"

namespace avio
{

class Reader
{
public:
	Reader() {}
	Reader(const char* filename);
	~Reader();
	AVPacket* read();

	void* process;

	void request_seek(float pct);
	int64_t seek_target_pts = AV_NOPTS_VALUE;
	int64_t seek_found_pts = AV_NOPTS_VALUE;
	int64_t stop_play_at_pts = AV_NOPTS_VALUE;
	int seek_stream_index();
	AVPacket* seek();
	bool seeking();
	void start_from(int milliseconds);
	void end_at(int milliseconds);

	int64_t start_time();
	int64_t duration();
	int64_t bit_rate();

	bool has_video();
	int width();
	int height();
	AVRational frame_rate();
	const char* str_pix_fmt();
	AVPixelFormat pix_fmt();
	const char* str_video_codec();
	AVCodecID video_codec();
	int64_t video_bit_rate();
	AVRational video_time_base();

	bool has_audio();
	int channels();
	int sample_rate();
	int frame_size();

	uint64_t channel_layout();
	std::string str_channel_layout();
	AVSampleFormat sample_format();
	const char* str_sample_format();
	AVCodecID audio_codec();
	const char* str_audio_codec();
	int64_t audio_bit_rate();
	AVRational audio_time_base();

	int keyframe_cache_size();
	bool request_pipe_write = false;
	bool pipe_out = false;
	bool pipe_out_enabled = false;
	std::string pipe_out_dir;
	std::string pipe_out_filename;

	AVFormatContext* fmt_ctx = NULL;
	int video_stream_index = -1;
	int audio_stream_index = -1;
	bool closed = false;
	int64_t last_video_pts = 0;
	int64_t last_audio_pts = 0;

	bool show_video_pkts = false;
	bool show_audio_pkts = false;

	std::string vpq_name;
	std::string apq_name;

	int vpq_max_size = 0;
	int apq_max_size = 0;

	std::string video_out() const { return std::string(vpq_name); }
	std::string audio_out() const { return std::string(apq_name); }
	void set_video_out(const std::string& name) { vpq_name = std::string(name); }
	void set_audio_out(const std::string& name) { apq_name = std::string(name); }

	bool request_break = false;
	bool running = false;
	std::string exit_error_msg;

	void clear_stream_queues();
	bool isPaused();
	void clear_decoders();
	void signal_eof();

	ExceptionHandler ex;
};

}

#endif // READER_H