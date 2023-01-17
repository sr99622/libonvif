/********************************************************************
* libavio/src/Frame.cpp
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

#include "Frame.h"

namespace avio
{

Frame::Frame() :
	m_frame(nullptr),
	m_rts(0)
{

}

Frame::Frame(const Frame& other) :
	m_rts(other.m_rts),
	m_frame(copyFrame(other.m_frame))
{

}

Frame::Frame(Frame&& other) noexcept :
	m_rts(other.m_rts)
{
	m_rts = other.m_rts;
	m_frame = other.m_frame;
	other.m_frame = nullptr;
}

Frame& Frame::operator=(const Frame& other)
{
	m_rts = other.m_rts;
	av_frame_free(&m_frame);
	m_frame = copyFrame(other.m_frame);
	return *this;
}

Frame& Frame::operator=(Frame&& other) noexcept
{
	m_rts = other.m_rts;
	m_frame = other.m_frame;
	other.m_frame = nullptr;

	return *this;
}

Frame::Frame(AVFrame* src)
{
	if (!m_frame) m_frame = av_frame_alloc();
	av_frame_move_ref(m_frame, src);
}

Frame::Frame(int width, int height, AVPixelFormat pix_fmt)
{
	m_rts = AV_NOPTS_VALUE;
	m_frame = av_frame_alloc();
	m_frame->width = width;
	m_frame->height = height;
	m_frame->format = pix_fmt;
	av_frame_get_buffer(m_frame, 0);
	av_frame_make_writable(m_frame);
	int data_size = width * height;

	switch (pix_fmt) {
	case AV_PIX_FMT_YUV420P:
		memset(m_frame->data[0], 0, data_size);
		memset(m_frame->data[1], 128, data_size >> 2);
		memset(m_frame->data[2], 128, data_size >> 2);
		break;
	case AV_PIX_FMT_BGR24:
		memset(m_frame->data[0], 0, data_size * 3);
		break;
	case AV_PIX_FMT_BGRA:
		memset(m_frame->data[0], 0, data_size * 4);
		break;
	default:
		std::cout << "Error: unsupported pix fmt" << std::endl;
	}
}

Frame::~Frame()
{
	av_frame_free(&m_frame);
}

void Frame::invalidate()
{
	av_frame_free(&m_frame);
	m_frame = nullptr;
	m_rts = 0;
}

AVFrame* Frame::copyFrame(AVFrame* src)
{
	if (!src)
		return nullptr;

	AVFrame* dst = av_frame_alloc();
	dst->format = src->format;
	dst->channel_layout = src->channel_layout;
	dst->sample_rate = src->sample_rate;
	dst->nb_samples = src->nb_samples;
	dst->channels = src->channels;
	dst->width = src->width;
	dst->height = src->height;
	av_frame_get_buffer(dst, 0);
	av_frame_make_writable(dst);
	av_frame_copy_props(dst, src);
	av_frame_copy(dst, src);

	return dst;
}

AVMediaType Frame::mediaType() const
{
	AVMediaType result = AVMEDIA_TYPE_UNKNOWN;
	if (isValid()) {
		if (m_frame->width > 0 && m_frame->height > 0)
			result = AVMEDIA_TYPE_VIDEO;
		else if (m_frame->nb_samples > 0 && m_frame->sample_rate > 0)
			result = AVMEDIA_TYPE_AUDIO;
	}
	return result;
}

std::string Frame::description() const
{
	std::stringstream str;
	if (isValid()) {
		if (mediaType() == AVMEDIA_TYPE_VIDEO) {
			const char* pix_fmt_name = av_get_pix_fmt_name((AVPixelFormat)m_frame->format);
			str << "VIDEO FRAME, width: " << m_frame->width << " height: " << m_frame->height
				<< " format: " << (pix_fmt_name ? pix_fmt_name : "unknown pixel format")
				<< " pts: " << m_frame->pts << " m_rts: " << m_rts;
		}
		else if (mediaType() == AVMEDIA_TYPE_AUDIO) {
			const char* sample_fmt_name = av_get_sample_fmt_name((AVSampleFormat)m_frame->format);
			char buf[256];
			av_get_channel_layout_string(buf, 256, m_frame->channels, m_frame->channel_layout);
			str << "AUDIO FRAME, nb_samples: " << m_frame->nb_samples << ", channels: " << m_frame->channels
				<< ", format: " << (sample_fmt_name ? sample_fmt_name : "unknown sample format")
				<< ", sample_rate: " << m_frame->sample_rate
				<< ", channel_layout: " << buf 
				<< ", extended_data: " << (m_frame->extended_data[0] ? "yes" : "no")
				<< ", pts: " << m_frame->pts << ", m_rts: " << m_rts;
		}
		else {
			str << "UNKNOWN MEDIA TYPE";
		}
	}
	else {
		str << "INVALID FRAME";
	}

	return str.str();
}

void Frame::set_rts(AVStream* stream)
{
	if (isValid()) {
		if (m_frame->pts == AV_NOPTS_VALUE) {
			m_rts = 0;
		}
		else {
			double factor = 1000 * av_q2d(stream->time_base);
			uint64_t start_time = (stream->start_time == AV_NOPTS_VALUE ? 0 : stream->start_time);
			m_rts = (pts() - start_time) * factor;
		}
	}
}

void Frame::set_pts(AVStream* stream)
{
	if (isValid()) {
		double factor = av_q2d(stream->time_base);
		m_frame->pts = m_rts / factor / 1000;
	}
}

}
