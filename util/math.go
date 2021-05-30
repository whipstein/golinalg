package util

import (
	"math"
)

const (
	// uvnan    = 0xFF800001
	// uvinf    = 0x7F800000
	// uvneginf = 0xFF800000
	// uvone    = 0x3F800000
	// mask     = 0xFF
	// shift = 32 - 8 - 1
	// bias     = 127
	// signMask = 1 << 31
	// fracMask = 1<<shift - 1
	epsf64 = 2.2204460492503131e-016
	tiny   = 2.2250738585072014e-308
	huge   = 1.7976931348623157e+308
	radix  = 2
	digits = 53
	minexp = -1021
	maxexp = 1024
)

func absint(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func maxf64(a ...float64) float64 {
	maxval := a[0]
	for _, val := range a {
		if val > maxval {
			maxval = val
		}
	}
	return maxval
}

func maxint(a ...int) int {
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

func minf64(a ...float64) float64 {
	minval := a[0]
	for _, val := range a {
		if val < minval {
			minval = val
		}
	}
	return minval
}

func minint(a ...int) int {
	minval := a[0]
	for _, val := range a {
		if val < minval {
			minval = val
		}
	}
	return minval
}

func signf64(a, b float64) float64 {
	return math.Copysign(a, b)
}
