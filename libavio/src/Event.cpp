/********************************************************************
* libavio/src/Event.cpp
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

#include "Event.h"

std::map<int, std::string> key_codes =
{
    { SDLK_0 , "SDLK_0" },
    { SDLK_1 , "SDLK_1" },
    { SDLK_2 , "SDLK_2" },
    { SDLK_3 , "SDLK_3" },
    { SDLK_4 , "SDLK_4" },
    { SDLK_5 , "SDLK_5" },
    { SDLK_6 , "SDLK_6" },
    { SDLK_7 , "SDLK_7" },
    { SDLK_8 , "SDLK_8" },
    { SDLK_9 , "SDLK_9" },
    { SDLK_a , "SDLK_a" },
    { SDLK_AC_BACK , "SDLK_AC_BACK" },
    { SDLK_AC_BOOKMARKS , "SDLK_AC_BOOKMARKS" },
    { SDLK_AC_FORWARD , "SDLK_AC_FORWARD" },
    { SDLK_AC_HOME , "SDLK_AC_HOME" },
    { SDLK_AC_REFRESH , "SDLK_AC_REFRESH" },
    { SDLK_AC_SEARCH , "SDLK_AC_SEARCH" },
    { SDLK_AC_STOP , "SDLK_AC_STOP" },
    { SDLK_AGAIN , "SDLK_AGAIN" },
    { SDLK_ALTERASE , "SDLK_ALTERASE" },
    { SDLK_QUOTE , "SDLK_QUOTE" },
    { SDLK_APPLICATION , "SDLK_APPLICATION" },
    { SDLK_AUDIOMUTE , "SDLK_AUDIOMUTE" },
    { SDLK_AUDIONEXT , "SDLK_AUDIONEXT" },
    { SDLK_AUDIOPLAY , "SDLK_AUDIOPLAY" },
    { SDLK_AUDIOPREV , "SDLK_AUDIOPREV" },
    { SDLK_AUDIOSTOP , "SDLK_AUDIOSTOP" },
    { SDLK_b , "SDLK_b" },
    { SDLK_BACKSLASH , "SDLK_BACKSLASH" },
    { SDLK_BACKSPACE , "SDLK_BACKSPACE" },
    { SDLK_BRIGHTNESSDOWN , "SDLK_BRIGHTNESSDOWN" },
    { SDLK_BRIGHTNESSUP , "SDLK_BRIGHTNESSUP" },
    { SDLK_c , "SDLK_c" },
    { SDLK_CALCULATOR , "SDLK_CALCULATOR" },
    { SDLK_CANCEL , "SDLK_CANCEL" },
    { SDLK_CAPSLOCK , "SDLK_CAPSLOCK" },
    { SDLK_CLEAR , "SDLK_CLEAR" },
    { SDLK_CLEARAGAIN , "SDLK_CLEARAGAIN" },
    { SDLK_COMMA , "SDLK_COMMA" },
    { SDLK_COMPUTER , "SDLK_COMPUTER" },
    { SDLK_COPY , "SDLK_COPY" },
    { SDLK_CRSEL , "SDLK_CRSEL" },
    { SDLK_CURRENCYSUBUNIT , "SDLK_CURRENCYSUBUNIT" },
    { SDLK_CURRENCYUNIT , "SDLK_CURRENCYUNIT" },
    { SDLK_CUT , "SDLK_CUT" },
    { SDLK_d , "SDLK_d" },
    { SDLK_DECIMALSEPARATOR , "SDLK_DECIMALSEPARATOR" },
    { SDLK_DELETE , "SDLK_DELETE" },
    { SDLK_DISPLAYSWITCH , "SDLK_DISPLAYSWITCH" },
    { SDLK_DOWN , "SDLK_DOWN" },
    { SDLK_e , "SDLK_e" },
    { SDLK_EJECT , "SDLK_EJECT" },
    { SDLK_END , "SDLK_END" },
    { SDLK_EQUALS , "SDLK_EQUALS" },
    { SDLK_ESCAPE , "SDLK_ESCAPE" },
    { SDLK_EXECUTE , "SDLK_EXECUTE" },
    { SDLK_EXSEL , "SDLK_EXSEL" },
    { SDLK_f , "SDLK_f" },
    { SDLK_F1 , "SDLK_F1" },
    { SDLK_F10 , "SDLK_F10" },
    { SDLK_F11 , "SDLK_F11" },
    { SDLK_F12 , "SDLK_F12" },
    { SDLK_F13 , "SDLK_F13" },
    { SDLK_F14 , "SDLK_F14" },
    { SDLK_F15 , "SDLK_F15" },
    { SDLK_F16 , "SDLK_F16" },
    { SDLK_F17 , "SDLK_F17" },
    { SDLK_F18 , "SDLK_F18" },
    { SDLK_F19 , "SDLK_F19" },
    { SDLK_F2 , "SDLK_F2" },
    { SDLK_F20 , "SDLK_F20" },
    { SDLK_F21 , "SDLK_F21" },
    { SDLK_F22 , "SDLK_F22" },
    { SDLK_F23 , "SDLK_F23" },
    { SDLK_F24 , "SDLK_F24" },
    { SDLK_F3 , "SDLK_F3" },
    { SDLK_F4 , "SDLK_F4" },
    { SDLK_F5 , "SDLK_F5" },
    { SDLK_F6 , "SDLK_F6" },
    { SDLK_F7 , "SDLK_F7" },
    { SDLK_F8 , "SDLK_F8" },
    { SDLK_F9 , "SDLK_F9" },
    { SDLK_FIND , "SDLK_FIND" },
    { SDLK_g , "SDLK_g" },
    { SDLK_BACKQUOTE , "SDLK_BACKQUOTE" },
    { SDLK_h , "SDLK_h" },
    { SDLK_HELP , "SDLK_HELP" },
    { SDLK_HOME , "SDLK_HOME" },
    { SDLK_i , "SDLK_i" },
    { SDLK_INSERT , "SDLK_INSERT" },
    { SDLK_j , "SDLK_j" },
    { SDLK_k , "SDLK_k" },
    { SDLK_KBDILLUMDOWN , "SDLK_KBDILLUMDOWN" },
    { SDLK_KBDILLUMTOGGLE , "SDLK_KBDILLUMTOGGLE" },
    { SDLK_KBDILLUMUP , "SDLK_KBDILLUMUP" },
    { SDLK_KP_0 , "SDLK_KP_0" },
    { SDLK_KP_00 , "SDLK_KP_00" },
    { SDLK_KP_000 , "SDLK_KP_000" },
    { SDLK_KP_1 , "SDLK_KP_1" },
    { SDLK_KP_2 , "SDLK_KP_2" },
    { SDLK_KP_3 , "SDLK_KP_3" },
    { SDLK_KP_4 , "SDLK_KP_4" },
    { SDLK_KP_5 , "SDLK_KP_5" },
    { SDLK_KP_6 , "SDLK_KP_6" },
    { SDLK_KP_7 , "SDLK_KP_7" },
    { SDLK_KP_8 , "SDLK_KP_8" },
    { SDLK_KP_9 , "SDLK_KP_9" },
    { SDLK_KP_A , "SDLK_KP_A" },
    { SDLK_KP_AMPERSAND , "SDLK_KP_AMPERSAND" },
    { SDLK_KP_AT , "SDLK_KP_AT" },
    { SDLK_KP_B , "SDLK_KP_B" },
    { SDLK_KP_BACKSPACE , "SDLK_KP_BACKSPACE" },
    { SDLK_KP_BINARY , "SDLK_KP_BINARY" },
    { SDLK_KP_C , "SDLK_KP_C" },
    { SDLK_KP_CLEAR , "SDLK_KP_CLEAR" },
    { SDLK_KP_CLEARENTRY , "SDLK_KP_CLEARENTRY" },
    { SDLK_KP_COLON , "SDLK_KP_COLON" },
    { SDLK_KP_COMMA , "SDLK_KP_COMMA" },
    { SDLK_KP_D , "SDLK_KP_D" },
    { SDLK_KP_DBLAMPERSAND , "SDLK_KP_DBLAMPERSAND" },
    { SDLK_KP_DECIMAL , "SDLK_KP_DECIMAL" },
    { SDLK_KP_DIVIDE , "SDLK_KP_DIVIDE" },
    { SDLK_KP_E , "SDLK_KP_E" },
    { SDLK_KP_ENTER , "SDLK_KP_ENTER" },
    { SDLK_KP_EQUALS , "SDLK_KP_EQUALS" },
    { SDLK_KP_EQUALSAS400 , "SDLK_KP_EQUALSAS400" },
    { SDLK_KP_EXCLAM , "SDLK_KP_EXCLAM" },
    { SDLK_KP_F , "SDLK_KP_F" },
    { SDLK_KP_GREATER , "SDLK_KP_GREATER" },
    { SDLK_KP_HASH , "SDLK_KP_HASH" },
    { SDLK_KP_HEXADECIMAL , "SDLK_KP_HEXADECIMAL" },
    { SDLK_KP_LEFTBRACE , "SDLK_KP_LEFTBRACE" },
    { SDLK_KP_LEFTPAREN , "SDLK_KP_LEFTPAREN" },
    { SDLK_KP_LESS , "SDLK_KP_LESS" },
    { SDLK_KP_MEMADD , "SDLK_KP_MEMADD" },
    { SDLK_KP_MEMCLEAR , "SDLK_KP_MEMCLEAR" },
    { SDLK_KP_MEMDIVIDE , "SDLK_KP_MEMDIVIDE" },
    { SDLK_KP_MEMMULTIPLY , "SDLK_KP_MEMMULTIPLY" },
    { SDLK_KP_MEMRECALL , "SDLK_KP_MEMRECALL" },
    { SDLK_KP_MEMSTORE , "SDLK_KP_MEMSTORE" },
    { SDLK_KP_MEMSUBTRACT , "SDLK_KP_MEMSUBTRACT" },
    { SDLK_KP_MINUS , "SDLK_KP_MINUS" },
    { SDLK_KP_MULTIPLY , "SDLK_KP_MULTIPLY" },
    { SDLK_KP_OCTAL , "SDLK_KP_OCTAL" },
    { SDLK_KP_PERCENT , "SDLK_KP_PERCENT" },
    { SDLK_KP_PERIOD , "SDLK_KP_PERIOD" },
    { SDLK_KP_PLUS , "SDLK_KP_PLUS" },
    { SDLK_KP_PLUSMINUS , "SDLK_KP_PLUSMINUS" },
    { SDLK_KP_POWER , "SDLK_KP_POWER" },
    { SDLK_KP_RIGHTBRACE , "SDLK_KP_RIGHTBRACE" },
    { SDLK_KP_RIGHTPAREN , "SDLK_KP_RIGHTPAREN" },
    { SDLK_KP_SPACE , "SDLK_KP_SPACE" },
    { SDLK_KP_TAB , "SDLK_KP_TAB" },
    { SDLK_KP_VERTICALBAR , "SDLK_KP_VERTICALBAR" },
    { SDLK_KP_XOR , "SDLK_KP_XOR" },
    { SDLK_l , "SDLK_l" },
    { SDLK_LALT , "SDLK_LALT" },
    { SDLK_LCTRL , "SDLK_LCTRL" },
    { SDLK_LEFT , "SDLK_LEFT" },
    { SDLK_LEFTBRACKET , "SDLK_LEFTBRACKET" },
    { SDLK_LGUI , "SDLK_LGUI" },
    { SDLK_LSHIFT , "SDLK_LSHIFT" },
    { SDLK_m , "SDLK_m" },
    { SDLK_MAIL , "SDLK_MAIL" },
    { SDLK_MEDIASELECT , "SDLK_MEDIASELECT" },
    { SDLK_MENU , "SDLK_MENU" },
    { SDLK_MINUS , "SDLK_MINUS" },
    { SDLK_MODE , "SDLK_MODE" },
    { SDLK_MUTE , "SDLK_MUTE" },
    { SDLK_n , "SDLK_n" },
    { SDLK_NUMLOCKCLEAR , "SDLK_NUMLOCKCLEAR" },
    { SDLK_o , "SDLK_o" },
    { SDLK_OPER , "SDLK_OPER" },
    { SDLK_OUT , "SDLK_OUT" },
    { SDLK_p , "SDLK_p" },
    { SDLK_PAGEDOWN , "SDLK_PAGEDOWN" },
    { SDLK_PAGEUP , "SDLK_PAGEUP" },
    { SDLK_PASTE , "SDLK_PASTE" },
    { SDLK_PAUSE , "SDLK_PAUSE" },
    { SDLK_PERIOD , "SDLK_PERIOD" },
    { SDLK_POWER , "SDLK_POWER" },
    { SDLK_PRINTSCREEN , "SDLK_PRINTSCREEN" },
    { SDLK_PRIOR , "SDLK_PRIOR" },
    { SDLK_q , "SDLK_q" },
    { SDLK_r , "SDLK_r" },
    { SDLK_RALT , "SDLK_RALT" },
    { SDLK_RCTRL , "SDLK_RCTRL" },
    { SDLK_RETURN , "SDLK_RETURN" },
    { SDLK_RETURN2 , "SDLK_RETURN2" },
    { SDLK_RGUI , "SDLK_RGUI" },
    { SDLK_RIGHT , "SDLK_RIGHT" },
    { SDLK_RIGHTBRACKET , "SDLK_RIGHTBRACKET" },
    { SDLK_RSHIFT , "SDLK_RSHIFT" },
    { SDLK_s , "SDLK_s" },
    { SDLK_SCROLLLOCK , "SDLK_SCROLLLOCK" },
    { SDLK_SELECT , "SDLK_SELECT" },
    { SDLK_SEMICOLON , "SDLK_SEMICOLON" },
    { SDLK_SEPARATOR , "SDLK_SEPARATOR" },
    { SDLK_SLASH , "SDLK_SLASH" },
    { SDLK_SLEEP , "SDLK_SLEEP" },
    { SDLK_SPACE , "SDLK_SPACE" },
    { SDLK_STOP , "SDLK_STOP" },
    { SDLK_SYSREQ , "SDLK_SYSREQ" },
    { SDLK_t , "SDLK_t" },
    { SDLK_TAB , "SDLK_TAB" },
    { SDLK_THOUSANDSSEPARATOR , "SDLK_THOUSANDSSEPARATOR" },
    { SDLK_u , "SDLK_u" },
    { SDLK_UNDO , "SDLK_UNDO" },
    { SDLK_UNKNOWN , "SDLK_UNKNOWN" },
    { SDLK_UP , "SDLK_UP" },
    { SDLK_v , "SDLK_v" },
    { SDLK_VOLUMEDOWN , "SDLK_VOLUMEDOWN" },
    { SDLK_VOLUMEUP , "SDLK_VOLUMEUP" },
    { SDLK_w , "SDLK_w" },
    { SDLK_WWW , "SDLK_WWW" },
    { SDLK_x , "SDLK_x" },
    { SDLK_y , "SDLK_y" },
    { SDLK_z , "SDLK_z" },
    { SDLK_AMPERSAND , "SDLK_AMPERSAND" },
    { SDLK_ASTERISK , "SDLK_ASTERISK" },
    { SDLK_AT , "SDLK_AT" },
    { SDLK_CARET , "SDLK_CARET" },
    { SDLK_COLON , "SDLK_COLON" },
    { SDLK_DOLLAR , "SDLK_DOLLAR" },
    { SDLK_EXCLAIM , "SDLK_EXCLAIM" },
    { SDLK_GREATER , "SDLK_GREATER" },
    { SDLK_HASH , "SDLK_HASH" },
    { SDLK_LEFTPAREN , "SDLK_LEFTPAREN" },
    { SDLK_LESS , "SDLK_LESS" },
    { SDLK_PERCENT , "SDLK_PERCENT" },
    { SDLK_PLUS , "SDLK_PLUS" },
    { SDLK_QUESTION , "SDLK_QUESTION" },
    { SDLK_QUOTEDBL , "SDLK_QUOTEDBL" },
    { SDLK_RIGHTPAREN , "SDLK_RIGHTPAREN" },
    { SDLK_UNDERSCORE , "SDLK_UNDERSCORE" },    
};

std::map<int, std::string> key_mods =
{
    { KMOD_NONE, "KMOD_NONE" },
    { KMOD_LSHIFT, "KMOD_LSHIFT" },
    { KMOD_RSHIFT, "KMOD_RSHIFT" },
    { KMOD_LCTRL, "KMOD_LCTRL" }, 
    { KMOD_RCTRL, "KMOD_RCTRL" },
    { KMOD_LALT, "KMOD_LALT" },
    { KMOD_RALT, "KMOD_RALT" },
    { KMOD_LGUI, "KMOD_LGUI" },
    { KMOD_RGUI, "KMOD_RGUI" },
    { KMOD_NUM, "KMOD_NUM" },
    { KMOD_CAPS, "KMOD_CAPS" },
    { KMOD_MODE, "KMOD_MODE" },
    { KMOD_CTRL, "KMOD_CTRL" },
    { KMOD_SHIFT, "KMOD_SHIFT" },
    { KMOD_ALT, "KMOD_ALT" },
    { KMOD_GUI, "KMOD_GUI" },
};

std::map<int, std::string> buttons = 
{
    { SDL_BUTTON_LEFT, "SDL_BUTTON_LEFT" },
    { SDL_BUTTON_MIDDLE, "SDL_BUTTON_MIDDLE" },
    { SDL_BUTTON_RIGHT, "SDL_BUTTON_RIGHT" },
    { SDL_BUTTON_X1, "SDL_BUTTON_X1" },
    { SDL_BUTTON_X2, "SDL_BUTTON_X2" },
};

std::map<int, std::string> mouse_state = 
{
    { SDL_BUTTON_LMASK, "SDL_BUTTON_LMASK" },
    { SDL_BUTTON_MMASK, "SDL_BUTTON_MMASK" },
    { SDL_BUTTON_RMASK, "SDL_BUTTON_RMASK" },
    { SDL_BUTTON_X1MASK, "SDL_BUTTON_X1MASK" },
    { SDL_BUTTON_X2MASK, "SDL_BUTTON_X2MASK" },
};


namespace avio
{

std::string Event::pack(const std::vector<SDL_Event>& events)
{
    std::stringstream str;
    for (const SDL_Event& e : events) {
        switch (e.type) {
            case SDL_KEYDOWN:
            str << "SDL_KEYDOWN," 
                << std::to_string(e.key.repeat) << ","
                << key_codes[e.key.keysym.sym] << ","
                << key_mods[e.key.keysym.mod] << ";";
            break;
            case SDL_KEYUP:
            str << "SDL_KEYUP," 
                << std::to_string(e.key.repeat) << ","
                << key_codes[e.key.keysym.sym] << ","
                << key_mods[e.key.keysym.mod] << ";";
            break;
            case SDL_MOUSEMOTION:
            str << "SDL_MOUSEMOTION,"
                << mouse_state[e.motion.state] << ","
                << std::to_string(e.motion.x) << ","
                << std::to_string(e.motion.y) << ";";
            break;
            case SDL_MOUSEBUTTONDOWN:
            str << "SDL_MOUSEBUTTONDOWN,"
                << buttons[e.button.button] << ","
                << std::to_string(e.button.clicks) << ","
                << std::to_string(e.button.x) << ","
                << std::to_string(e.button.y) << ";";
            break;
            case SDL_MOUSEBUTTONUP:
            str << "SDL_MOUSEBUTTONUP,"
                << buttons[e.button.button] << ","
                << std::to_string(e.button.clicks) << ","
                << std::to_string(e.button.x) << ","
                << std::to_string(e.button.y) << ";";
            break;
        }
    }
    return str.str();
}

}