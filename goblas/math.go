package goblas

import (
	"math"
)

const (
	uvnan    = 0xFF800001
	uvinf    = 0x7F800000
	uvneginf = 0xFF800000
	uvone    = 0x3F800000
	mask     = 0xFF
	shift    = 32 - 8 - 1
	bias     = 127
	signMask = 1 << 31
	fracMask = 1<<shift - 1
	eps      = 2.2204460492503131e-016
)

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func abssumc128(a complex128) float64 {
	return (math.Abs(real(a)) + math.Abs(imag(a)))
}

func epsilonf32() float32 {
	return float32(math.Pow(2, -23))
	// return float32(1.1920929e-07)
}

func epsilonf64() float64 {
	return math.Pow(2, -52)
	// return float32(1.1920929e-07)
}

// func math.Max(a ...float64) float64 {
// 	maxval := a[0]
// 	for _, val := range a {
// 		if val > maxval {
// 			maxval = val
// 		}
// 	}
// 	return maxval
// }

func max(a ...int) int {
	maxval := a[0]
	for _, val := range a {
		if val > maxval {
			maxval = val
		}
	}
	return maxval
}

func maxslice(a []int) int {
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

// func math.Min(a ...float64) float64 {
// 	minval := a[0]
// 	for _, val := range a {
// 		if val < minval {
// 			minval = val
// 		}
// 	}
// 	return minval
// }

func min(a ...int) int {
	minval := a[0]
	for _, val := range a {
		if val < minval {
			minval = val
		}
	}
	return minval
}

func sign(a, b int) int {
	if b < 0 && a > 0 {
		return -a
	}
	return a
}
