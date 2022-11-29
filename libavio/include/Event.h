/********************************************************************
* libavio/include/Event.h
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

#ifndef EVENT_H
#define EVENT_H

#include <SDL.h>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace avio 
{

class Event
{
public:
    Event() {}

    static std::string pack(const std::vector<SDL_Event>& events);
};

}

#endif // EVENT_H