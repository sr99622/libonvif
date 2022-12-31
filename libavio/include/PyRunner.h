/********************************************************************
* libavio/include/PyRunner.h
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

#ifndef PYRUNNER_H
#define PYRUNNER_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL AV_ARRAY_API
#include "numpy/arrayobject.h"
#define NO_IMPORT_ARRAY

#include <functional>
#include <iostream>

#include "Frame.h"
#include "Exception.h"

namespace avio
{

class PyRunner {
public:
	PyRunner();
	~PyRunner();
	int initialize(const std::string& dir, const std::string& file, const std::string& py_class, const std::string& arg);
	PyObject* getImage(Frame& f);
	bool run(Frame& f);

	PyObject* pClass = nullptr;
	ExceptionHandler ex;
    uint8_t* mat_buf = nullptr;
	PyObject* pData = NULL;

};

}

#endif // PYRUNNER_H