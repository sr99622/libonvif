/********************************************************************
* libavio/include/Filter.h
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

#ifndef FILTER_H
#define FILTER_H

extern "C" {
#include "libavcodec/avfft.h"
#include "libavfilter/avfilter.h"
#include "libswresample/swresample.h"
#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
#include "libavformat/avformat.h"
}

#include "Exception.h"
#include "Decoder.h"

namespace avio
{

class Filter
{
public: 
	Filter(Decoder& decoder, const char* description);
	void initVideo();
	void initAudio();
	~Filter();
	void filter(Frame& f);
	AVMediaType mediaType() { return decoder->mediaType; }

	Decoder* decoder = NULL;
	AVFilterContext* sink_ctx = NULL;
	AVFilterContext* src_ctx = NULL;
	AVFilterGraph* graph = NULL;
	AVFrame* frame = NULL;
	Frame tmp;
	std::string desc;

	uint64_t m_channel_layout = 0;

	int width() { return av_buffersink_get_w(sink_ctx); }
	int height() { return av_buffersink_get_h(sink_ctx); }
	AVPixelFormat pix_fmt() { return (AVPixelFormat)av_buffersink_get_format(sink_ctx); }
	AVRational time_base() { return av_buffersink_get_time_base(sink_ctx); }
	AVRational frame_rate() { return av_buffersink_get_frame_rate(sink_ctx); }
	int sample_rate() { return av_buffersink_get_sample_rate(sink_ctx); }
	int channels() { return av_buffersink_get_channels(sink_ctx); }
	int64_t channel_layout() { return av_buffersink_get_channel_layout(sink_ctx); }
	AVSampleFormat sample_format() { return (AVSampleFormat)av_buffersink_get_format(sink_ctx); }
	int frame_size() { return decoder->frame_size(); }

	Queue<Frame>* frame_out_q = nullptr;
	std::string q_in_name;
	std::string q_out_name;

	bool show_frames = false;

	std::string video_in() const { return std::string(q_in_name); }
	std::string video_out() const { return std::string(q_out_name); }
	std::string audio_in() const { return std::string(q_in_name); }
	std::string audio_out() const { return std::string(q_out_name); }
	void set_audio_in(const std::string& name) { q_in_name = std::string(name); }
	void set_audio_out(const std::string& name) { q_out_name = std::string(name); }
	void set_video_in(const std::string& name) { q_in_name = std::string(name); }
	void set_video_out(const std::string& name) { q_out_name = std::string(name); }


	ExceptionHandler ex;
};

}

#endif // FILTER_H