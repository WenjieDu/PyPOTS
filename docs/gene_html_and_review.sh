# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

# This shell script is created to help generate PyPOTS html documentation for review before publishing.

make html && \
  python -m http.server 8099 -d _build/html