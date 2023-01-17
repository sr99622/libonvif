/********************************************************************
* libavio/include/Frame.h
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

#ifndef FRAME_H
#define FRAME_H

extern "C" {
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
}

#include <iostream>
#include <sstream>

namespace avio
{

class Frame
{
public:
	Frame();
	~Frame();
	Frame(const Frame& other);
	Frame(Frame&& other) noexcept;
	Frame(AVFrame* src);
	Frame(int width, int height, AVPixelFormat pix_fmt);
	Frame& operator=(const Frame& other);
	Frame& operator=(Frame&& other) noexcept;
	AVFrame* copyFrame(AVFrame* src);
	bool isValid() const { return m_frame ? true : false; }
	void invalidate();
	AVMediaType mediaType() const;
	uint64_t pts() { return m_frame ? m_frame->pts : AV_NOPTS_VALUE; }
	void set_rts(AVStream* stream);  // called from Decoder::decode
	void set_pts(AVStream* stream);  // called from Encoder::encode
	std::string description() const;

	AVFrame* m_frame = nullptr;
	uint64_t m_rts;
	bool m_faded = false;  // used by osd to avoid duplicate background fade

};

}

#endif // FRAME_H