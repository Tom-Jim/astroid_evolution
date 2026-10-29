# Cached runs

`.aecache` files are stored timelines of viewer runs (`src/app/cache.rs`). The web page
looks here for the file of the run it opens (`cache/<project>-<key>.aecache`, the name
the EXPORT CACHE button gives) and plays it instead of computing; `?live` computes
anyway. The desktop app computes first and writes exports here when run from the
source tree. Any `.aecache` file can be played with IMPORT CACHE.
