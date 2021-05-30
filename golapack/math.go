package golapack

import (
	"math"
	"math/cmplx"
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

// func absf32(a float32) float32 {
// 	if a < 0 {
// 		return -a
// 	}
// 	return a
// }

// func absc64(a complex64) float32 {
// 	return float32(cmplx.Abs(complex128(a)))
// }

func absint(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// func abssumc64(a complex64) float32 {
// 	return float32(absf32(real(a)) + absf32(imag(a)))
// }

// func abssumc128(a complex128) float64 {
// 	return (math.Abs(real(a)) + math.Abs(imag(a)))
// }

// func ceilingf32(a float32) int {
// 	return int(math.Ceil(float64(a)))
// }

// func conjc64(a complex64) complex64 {
// 	return complex64(cmplx.Conj(complex128(a)))
// }

// func cosf32(a float32) float32 {
// 	return float32(math.Cos(float64(a)))
// }

// func epsilonf32() float32 {
// 	return powf32(2, -23)
// 	// return float32(1.1920929e-07)
// }

// func epsilonf64() float64 {
// 	return math.Pow(2, -52)
// 	// return float32(1.1920929e-07)
// }

// func expf32(a float32) float32 {
// 	return float32(math.Exp(float64(a)))
// }

// func logf32(a float32) float32 {
// 	return float32(math.Log(float64(a)))
// }

// func log10f32(a float32) float32 {
// 	return float32(math.Log10(float64(a)))
// }

// func maxf32(a ...float32) float32 {
// 	maxval := a[0]
// 	for _, val := range a {
// 		if val > maxval {
// 			maxval = val
// 		}
// 	}
// 	return maxval
// }

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

func maxlocc128(a ...complex128) int {
	var val float64

	maxval := cmplx.Abs(a[0])
	maxloc := 1
	for i, cval := range a {
		val = cmplx.Abs(cval)
		if val > maxval {
			maxval = val
			maxloc = i
		}
	}
	return maxloc + 1
}

// func minf32(a ...float32) float32 {
// 	minval := a[0]
// 	for _, val := range a {
// 		if val < minval {
// 			minval = val
// 		}
// 	}
// 	return minval
// }

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

// func minintslice(a []int) int {
// 	var out int

// 	for _, val := range a {
// 		if val < out {
// 			out = val
// 		}
// 	}
// 	return out
// }

// func nintf32(a float32) int {
// 	return int(math.Round(float64(a)))
// }

// func powf32(a, b float32) float32 {
// 	return float32(math.Pow(float64(a), float64(b)))
// }

func powint(a, b int) int {
	if b == 0 {
		return 1
	}

	result := a
	for i := 1; i < b; i++ {
		result *= a
	}
	return result
}

// func roundf32(a float32) float32 {
// 	var sd int = 8 // significant digits of a float32
// 	sc := float32(math.Pow10(sd))
// 	b := int(a * sc)
// 	return float32(b) / sc
// }

// func signf32(a, b float32) float32 {
// 	return float32(math.Copysign(float64(a), float64(b)))
// }

func signf64(a, b float64) float64 {
	return math.Copysign(a, b)
}

// func signint(a, b int) int {
// 	return int(math.Copysign(float64(a), float64(b)))
// }

// func sinf32(a float32) float32 {
// 	return float32(math.Sin(float64(a)))
// }

// func sqrtf32(a float32) float32 {
// 	return float32(math.Sqrt(float64(a)))
// }
