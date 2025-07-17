# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#FROM node:18-alpine

FROM nvcr.io/nvidia/base/ubuntu:22.04_20240212

RUN apt update
RUN apt install -y \
    apt-transport-https \
    curl \
    software-properties-common

RUN curl -sL https://deb.nodesource.com/setup_21.x | bash -
RUN apt install -y nodejs

WORKDIR /app

COPY . .
RUN ln -s /app/shared/images /app/public/images

RUN npm install

CMD ["npm", "run", "start"]
