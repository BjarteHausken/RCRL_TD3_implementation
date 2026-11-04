# Third-party notices

This repository contains code adapted from the project below. The license
reproduced here applies to those adapted portions.

## TD3 reference implementation

- Source: <https://github.com/sfujim/TD3>
- Adapted in: `agents/td3.py`, the `ReplayBuffer` class in
  `utils/replay_buffer.py`, and the TD3 training loops in
  `train_scripts/train_td3.py` and `train_scripts/train_td3_vec.py`.
  `agents/rcrl_td3.py` builds on the same actor and critic structure.

```text
MIT License

Copyright (c) 2020 Scott Fujimoto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
