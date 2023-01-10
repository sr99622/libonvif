/********************************************************************
* libavio/include/Display.h
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

#ifndef DISPLAY_H
#define DISPLAY_H

#define SDL_MAIN_HANDLED

#include <SDL.h>
#include <chrono>
#include <deque>
#include "Exception.h"
#include "Queue.h"
#include "Frame.h"
#include "Clock.h"
#include "Decoder.h"
#include "Filter.h"
#include "Reader.h"
#include "Writer.h"
#include "Encoder.h"

#define SDL_EVENT_LOOP_WAIT 10

namespace avio
{

enum class PlayState {
    PLAY,
    PAUSE,
    QUIT
};

class Display
{

public:
    Display(Reader& reader) : reader(&reader) { }
    ~Display();

    void* process;
    Reader* reader;

    void init();
    int initAudio(int sample_rate, AVSampleFormat sample_fmt, int channels, uint64_t channel_layout, int stream_nb_samples);
    int initVideo(int width, int height, AVPixelFormat pix_fmt);
    static void AudioCallback(void* userdata, uint8_t* stream, int len);
    void videoPresentation();
    PlayState getEvents(std::vector<SDL_Event>* events);
    bool display();
    void pin_osd(bool arg);
    void enable_status(bool arg);
    void snapshot();
    
    bool paused = false;
    Frame paused_frame;
    bool isPaused();
    void togglePause();
    bool single_step = false;
    bool reverse_step = false;
    int recent_idx = -1;

    bool recording = false;
    void toggleRecord();

    Frame f;

    std::string audioDeviceStatus() const;
    const char* sdlAudioFormatName(SDL_AudioFormat format) const;

    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;
    SDL_Texture* texture = NULL;
    SDL_Surface* screen = NULL;
    SDL_AudioSpec sdl = { 0 };
    SDL_AudioSpec have = { 0 };

    std::string vfq_in_name;
    std::string afq_in_name;
    std::string vfq_out_name;
    std::string afq_out_name;

    Queue<Frame>* vfq_in = nullptr;
    Queue<Frame>* afq_in = nullptr;
    Queue<Frame>* vfq_out = nullptr;
    Queue<Frame>* afq_out = nullptr;

    std::string video_in() const { return std::string(vfq_in_name); }
    std::string audio_in() const { return std::string(afq_in_name); }
    std::string video_out() const { return std::string(vfq_out_name); }
    std::string audio_out() const { return std::string(afq_out_name); }
    void set_video_in(const std::string& name) { vfq_in_name = std::string(name); }
    void set_audio_in(const std::string& name) { afq_in_name = std::string(name); }
    void set_video_out(const std::string& name) { vfq_out_name = std::string(name); }
    void set_audio_out(const std::string& name) { afq_out_name = std::string(name); }

    bool fullscreen = false;

    SDL_AudioDeviceID audioDeviceID;
    Clock rtClock;

    SwrContext* swr_ctx = nullptr;
    AVSampleFormat sdl_sample_format = AV_SAMPLE_FMT_S16;

    uint8_t* swr_buffer = nullptr;
    int swr_buffer_size = 0;
    int audio_buffer_len = 0;
    bool audio_eof = false;
    float volume = 1.0f;
    bool mute = false;

    int width = 0;
    int height = 0;
    AVPixelFormat pix_fmt = AV_PIX_FMT_NONE;

    uint64_t start_time;
    uint64_t duration;

    std::chrono::steady_clock clock;

    std::deque<Frame> recent;
    bool request_recent_clear = false;
    int recent_q_size = 200;
    bool prepend_recent_write = false;

    ExceptionHandler ex;

};

}

#endif // DISPLAY_H