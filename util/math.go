package util

const (
	epsf64 = 2.2204460492503131e-016
	tiny   = 2.2250738585072014e-308
	huge   = 1.7976931348623157e+308
	radix  = 2
	digits = 53
	minexp = -1021
	maxexp = 1024
)

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func max(a ...int) int {
	maxval := a[0]
	for _, val := range a {
		if val > maxval {
			maxval = val
		}
	}
	return maxval
}

func maxintslice(a []int) int {
	var out int

	for _, val := range a {
		if val > out {
			out = val
		}
	}
	return out
}

func maxlocf64(a ...float64) int {
	maxval := a[0]
	maxloc := 1
	for i, val := range a {
		if val > maxval {
			maxval = val
			maxloc = i
		}
	}
	return maxloc + 1
}

func min(a ...int) int {
	minval := a[0]
	for _, val := range a {
		if val < minval {
			minval = val
		}
	}
	return minval
}
