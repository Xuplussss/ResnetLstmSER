# data format

## tag / label
- TextGrid file (python tgt)
### example:
```
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 3.7333333333333334
tiers? <exists>
size = 1
item []:
	item [1]:
		class = "IntervalTier"
		name = "silences"
		xmin = 0.0
		xmax = 3.7333333333333334
		intervals: size = 4
		intervals [1]:
			xmin = 0.0
			xmax = 2.554666666666667
			text = "44"
		intervals [2]:
			xmin = 2.554666666666667
			xmax = 3.0026666666666664
			text = "37"
		intervals [3]:
			xmin = 3.0026666666666664
			xmax = 3.4106666666666667
			text = "44"
		intervals [4]:
			xmin = 3.4106666666666667
			xmax = 3.7333333333333334
			text = "37"
```
This audio contains four segments(intervals), where xmin and xmax mean the start point and end point of the segment. We labeled the segmented labels in the "text". Where the labels contains two characters, the first is the sound type and the second is the emotion.
## data
- wav file
- mono
- 16k
