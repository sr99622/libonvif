/********************************************************************
* libavio/src/Filter.cpp
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

#include "../include/Filter.h"
#include <SDL.h>

namespace avio
{

Filter::Filter(Decoder& decoder, const char* description) : decoder(&decoder), desc(description)
{
    switch (decoder.mediaType) {
    case AVMEDIA_TYPE_VIDEO:
        initVideo();
        break;
    case AVMEDIA_TYPE_AUDIO:
        initAudio();
        break;
    default:
        std::cout << "Filter constructor error, unknown media type" << std::endl;
        break;
    }
}

void Filter::initVideo()
{
    const AVFilter* buffersrc = avfilter_get_by_name("buffer");
    const AVFilter* buffersink = avfilter_get_by_name("buffersink");
    AVFilterInOut* outputs = avfilter_inout_alloc();
    AVFilterInOut* inputs = avfilter_inout_alloc();

    char args[512];
    snprintf(args, sizeof(args),
        "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
        decoder->dec_ctx->width, decoder->dec_ctx->height, decoder->dec_ctx->pix_fmt,
        decoder->stream->time_base.num, decoder->stream->time_base.den,
        decoder->dec_ctx->sample_aspect_ratio.num, decoder->dec_ctx->sample_aspect_ratio.den);

    try {
        ex.ck(frame = av_frame_alloc(), AFA);
        ex.ck(graph = avfilter_graph_alloc(), AGA);
        ex.ck(avfilter_graph_create_filter(&src_ctx, buffersrc, "in", args, NULL, graph), AGCF);
        ex.ck(avfilter_graph_create_filter(&sink_ctx, buffersink, "out", NULL, NULL, graph), AGCF);

        outputs->name = av_strdup("in");
        outputs->filter_ctx = src_ctx;
        outputs->pad_idx = 0;
        outputs->next = NULL;

        inputs->name = av_strdup("out");
        inputs->filter_ctx = sink_ctx;
        inputs->pad_idx = 0;
        inputs->next = NULL;

        ex.ck(avfilter_graph_parse_ptr(graph, desc.c_str(), &inputs, &outputs, NULL), AGPP);
        ex.ck(avfilter_graph_config(graph, NULL), AGC);
    }
    catch (const Exception& e) {
        ex.msg(e.what(), MsgPriority::CRITICAL, "Video Filter constructor exception: ");
    }

    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
}

void Filter::initAudio()
{
    AVFilterInOut* outputs = avfilter_inout_alloc();
    AVFilterInOut* inputs = avfilter_inout_alloc();
    const AVFilter* buf_src = avfilter_get_by_name("abuffer");
    const AVFilter* buf_sink = avfilter_get_by_name("abuffersink");

    //static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_U8, AV_SAMPLE_FMT_NONE };
    //static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_NONE };
    static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_NONE };

    try {
        if (decoder->dec_ctx->channel_layout && av_get_channel_layout_nb_channels(decoder->dec_ctx->channel_layout) == decoder->dec_ctx->channels)
            m_channel_layout = decoder->dec_ctx->channel_layout;

        std::stringstream str;
        str << "sample_rate=" << decoder->dec_ctx->sample_rate << ":"
            << "sample_fmt=" << av_get_sample_fmt_name(decoder->dec_ctx->sample_fmt) << ":"
            << "channels=" << decoder->dec_ctx->channels << ":"
            << "time_base=" << decoder->stream->time_base.num << "/" << decoder->stream->time_base.den;

        if (m_channel_layout)
            str << ":channel_layout=0x" << std::hex << m_channel_layout;

        ex.ck(frame = av_frame_alloc(), AFA);
        ex.ck(graph = avfilter_graph_alloc(), AGA);
        ex.ck(avfilter_graph_create_filter(&src_ctx, buf_src, "buf_src", str.str().c_str(), NULL, graph), AGCF);
        ex.ck(avfilter_graph_create_filter(&sink_ctx, buf_sink, "buf_sink", NULL, NULL, graph), AGCF);
        ex.ck(av_opt_set_int_list(sink_ctx, "sample_fmts", sample_fmts, AV_SAMPLE_FMT_NONE, AV_OPT_SEARCH_CHILDREN), AOSIL);
        ex.ck(av_opt_set_int(sink_ctx, "all_channel_counts", 1, AV_OPT_SEARCH_CHILDREN), AOSI);

        if (desc.c_str()) {
            if (!outputs || !inputs) throw Exception("avfilter_inout_alloc");

            outputs->name = av_strdup("in");
            outputs->filter_ctx = src_ctx;
            outputs->pad_idx = 0;
            outputs->next = NULL;

            inputs->name = av_strdup("out");
            inputs->filter_ctx = sink_ctx;
            inputs->pad_idx = 0;
            inputs->next = NULL;

            ex.ck(avfilter_graph_parse_ptr(graph, desc.c_str(), &inputs, &outputs, NULL), AGPP);
        }
        else {
            ex.ck(avfilter_link(src_ctx, 0, sink_ctx, 0), AL);
        }

        ex.ck(avfilter_graph_config(graph, NULL), AGC);
    }
    catch (const Exception& e) {
        ex.msg(e.what(), MsgPriority::CRITICAL, "Audio Filter constructor exception: ");
    }

    avfilter_inout_free(&outputs);
    avfilter_inout_free(&inputs);
}

Filter::~Filter()
{
    avfilter_free(sink_ctx);
    avfilter_free(src_ctx);
	avfilter_graph_free(&graph);
	av_frame_free(&frame);
}

void Filter::filter(const Frame& f)
{
    int ret = 0;

    try {
        ex.ck(av_buffersrc_add_frame_flags(src_ctx, f.m_frame, AV_BUFFERSRC_FLAG_KEEP_REF), ABAFF);
        while (true) {
            ret = av_buffersink_get_frame(sink_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0)
                throw Exception("av_buffersink_get_frame");

            tmp = Frame(frame);
            tmp.m_rts = tmp.pts() * 1000 * av_q2d(av_buffersink_get_time_base(sink_ctx));
            if (show_frames) std::cout << "filter " << f.description() << std::endl;

            frame_out_q->push(tmp);
            av_frame_unref(frame);
        }
    }
    catch (const Exception& e) {
        ex.msg(e.what(), MsgPriority::CRITICAL, "VideoFilter::filter exception: ");
    }
}

}
