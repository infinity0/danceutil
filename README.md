# Dance moves utility

## Usage

~~~~
# Print a table of moves, sorted by "from" position
$ ./danceutil.py tab

# Generate moves for a 128-count 32-bar AABA dance, from closed position
$ ./danceutil.py gen dance128 -n2 --fr closed/2H

# Generate moves for a 32-count phrase
$ ./danceutil.py gen phrases32 -n8

# More knobs to play with
$ ./danceutil.py -h
$ ./danceutil.py tab -h
$ ./danceutil.py gen -h
~~~~

## How to read the output

Positions are notated like `closed/2H`, and what these all mean can be read from `positions.yml`.

Every move has several possible from/to position pairs (a "transition") and takes a certain number of counts. When selected as part of a move sequence, one specific from/to transition is chosen, and this is shown in the output of the move sequence.

The full list of moves are a WIP and can be read in `swing-moves.txt`.

For more details, RTFS or ask me IRL.
