# MoccAI

A simple machine learning demo to detect coffee in a Moccamaster brand coffee makers (which we just happen to have in our office at Evermade).

## Usage

1. Install Python3.7+
2. Install pipenv
3. Run `pipenv install`
4. Run `pipenv run python3 main.py`

## Camera config

`v4l2-ctl -d /dev/video0 -c exposure_auto=1 -c exposure_auto_priority=0 -c exposure_absolute=2`

## License

TBD
