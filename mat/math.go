package mat

import (
	"math"
	"math/cmplx"
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
)

// func abs(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return float32(math.Abs(x.Float()))
// 	case float32:
// 		return float32(math.Abs(aval.Float()))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return math.Abs(x.Float())
// 	case float64:
// 		return math.Abs(aval.Float())
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return float32(cmplx.Abs(x.Complex()))
// 	case complex64:
// 		return float32(cmplx.Abs(aval.Complex()))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return cmplx.Abs(x.Complex())
// 	case complex128:
// 		return cmplx.Abs(aval.Complex())
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		if x.Int() < 0 {
// 			return int(-x.Int())
// 		}
// 		return int(x.Int())
// 	case int:
// 		if aval.Int() < 0 {
// 			return int(-aval.Int())
// 		}
// 		return int(aval.Int())
// 	default:
// 		log.Panic("abs: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func absf32(a float32) float32 {
	if a < 0 {
		return -a
	}
	return a
}

func absf64(a float64) float64 {
	if a < 0 {
		return -a
	}
	return a
}

func absc64(a complex64) float32 {
	return float32(cmplx.Abs(complex128(a)))
}

func absc128(a complex128) float64 {
	return cmplx.Abs(a)
}

func absint(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// func abssum(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return float32(math.Abs(real(x.Complex())) + math.Abs(imag(x.Complex())))
// 	case complex64:
// 		return float32(math.Abs(real(aval.Complex())) + math.Abs(imag(aval.Complex())))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return math.Abs(real(x.Complex())) + math.Abs(imag(x.Complex()))
// 	case complex128:
// 		return math.Abs(real(aval.Complex())) + math.Abs(imag(aval.Complex()))
// 	default:
// 		log.Panic("abssum: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func abssumc64(a complex64) float32 {
	return float32(absf32(real(a)) + absf32(imag(a)))
}

func abssumc128(a complex128) float64 {
	return (absf64(real(a)) + absf64(imag(a)))
}

// func ceiling(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return int(math.Ceil(x.Float()))
// 	case float32:
// 		return int(math.Ceil(aval.Float()))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return int(math.Ceil(x.Float()))
// 	case float64:
// 		return math.Ceil(aval.Float())
// 	default:
// 		log.Panic("ceiling: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func ceilingf32(a float32) int {
	return int(math.Ceil(float64(a)))
}

// func conj(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return complex64(cmplx.Conj(x.Complex()))
// 	case complex64:
// 		return complex64(cmplx.Conj(aval.Complex()))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return cmplx.Conj(x.Complex())
// 	case complex128:
// 		return cmplx.Conj(aval.Complex())
// 	default:
// 		log.Panic("conj: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

// func complx(a, b interface{}) interface{} {
// 	var bout float64

// 	aval := reflect.ValueOf(a)
// 	bval := reflect.ValueOf(b)

// 	switch b.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(bval)
// 		bout = x.Float()
// 	case float32:
// 		bout = bval.Float()
// 	case *float64:
// 		x := reflect.Indirect(bval)
// 		bout = x.Float()
// 	case float64:
// 		bout = bval.Float()
// 	default:
// 		log.Panic("complx: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}

// 	switch b.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return complex64(complex(x.Float(), bout))
// 	case float32:
// 		return complex64(complex(aval.Float(), bout))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return complex(x.Float(), bout)
// 	case float64:
// 		return complex(aval.Float(), bout)
// 	default:
// 		log.Panic("complx: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}

// 	return nil
// }

func conjc64(a complex64) complex64 {
	return complex64(cmplx.Conj(complex128(a)))
}

func conjc128(a complex128) complex128 {
	return cmplx.Conj(a)
}

// func cos(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return float32(math.Cos(x.Float()))
// 	case float32:
// 		return float32(math.Cos(aval.Float()))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return math.Cos(x.Float())
// 	case float64:
// 		return math.Cos(aval.Float())
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return complex64(cmplx.Cos(x.Complex()))
// 	case complex64:
// 		return complex64(cmplx.Cos(aval.Complex()))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return cmplx.Cos(x.Complex())
// 	case complex128:
// 		return cmplx.Cos(aval.Complex())
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		return int(math.Cos(float64(x.Int())))
// 	case int:
// 		return int(math.Cos(float64(aval.Int())))
// 	default:
// 		log.Panic("cos: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func cosf32(a float32) float32 {
	return float32(math.Cos(float64(a)))
}

// func epsilon(a interface{}) interface{} {
// 	switch a.(_type) {
// 	case *float32:
// 		return float32(1.1920929e-07)
// 	case float32:
// 		return float32(powerf32(2, -23))
// 		// return float32(1.1920929e-07)
// 	case *float64:
// 		return 0.22204460e-15
// 	case float64:
// 		return 0.22204460e-15
// 	default:
// 		log.Panic("epsilon: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func epsilonf32() float32 {
	return powf32(2, -23)
	// return float32(1.1920929e-07)
}

func epsilonf64() float64 {
	return powf64(2, -52)
	// return float32(1.1920929e-07)
}

// func exp(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return float32(math.Exp(x.Float()))
// 	case float32:
// 		return float32(math.Exp(aval.Float()))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return math.Exp(x.Float())
// 	case float64:
// 		return math.Exp(aval.Float())
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return complex64(cmplx.Exp(x.Complex()))
// 	case complex64:
// 		return complex64(cmplx.Exp(aval.Complex()))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return cmplx.Exp(x.Complex())
// 	case complex128:
// 		return cmplx.Exp(aval.Complex())
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		return int(math.Exp(float64(x.Int())))
// 	case int:
// 		return int(math.Exp(float64(aval.Int())))
// 	default:
// 		log.Panic("exp: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func expf32(a float32) float32 {
	return float32(math.Exp(float64(a)))
}

// func im(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return float32(imag(x.Complex()))
// 	case complex64:
// 		return float32(imag(aval.Complex()))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return imag(x.Complex())
// 	case complex128:
// 		return imag(aval.Complex())
// 	default:
// 		log.Panic("im: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

// func logarithm(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return float32(math.Log(x.Float()))
// 	case float32:
// 		return float32(math.Log(aval.Float()))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return math.Log(x.Float())
// 	case float64:
// 		return math.Log(aval.Float())
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return complex64(cmplx.Log(x.Complex()))
// 	case complex64:
// 		return complex64(cmplx.Log(aval.Complex()))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return cmplx.Log(x.Complex())
// 	case complex128:
// 		return cmplx.Log(aval.Complex())
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		return int(math.Log(float64(x.Int())))
// 	case int:
// 		return int(math.Log(float64(aval.Int())))
// 	default:
// 		log.Panic("logarithm: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func logf32(a float32) float32 {
	return float32(math.Log(float64(a)))
}

// func log10(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return float32(math.Log10(x.Float()))
// 	case float32:
// 		return float32(math.Log10(aval.Float()))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return math.Log10(x.Float())
// 	case float64:
// 		return math.Log10(aval.Float())
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return complex64(cmplx.Log10(x.Complex()))
// 	case complex64:
// 		return complex64(cmplx.Log10(aval.Complex()))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return cmplx.Log10(x.Complex())
// 	case complex128:
// 		return cmplx.Log10(aval.Complex())
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		return int(math.Log10(float64(x.Int())))
// 	case int:
// 		return int(math.Log10(float64(aval.Int())))
// 	default:
// 		log.Panic("log10: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func log10f32(a float32) float32 {
	return float32(math.Log10(float64(a)))
}

// func maxint(a, b interface{}) interface{} {
// 	aval := reflect.ValueOf(a)
// 	bval := reflect.ValueOf(b)

// 	if reflect.TypeOf(a) != reflect.TypeOf(b) {
// 		log.Panic("maxint: mismatched data types: ", reflect.TypeOf(a), reflect.TypeOf(b))
// 	}

// 	switch a.(_type) {
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		y := reflect.Indirect(bval)
// 		if x.Int() >= y.Int() {
// 			return int(x.Int())
// 		}
// 		return int(y.Int())
// 	case int:
// 		if aval.Int() >= bval.Int() {
// 			return int(aval.Int())
// 		}
// 		return int(bval.Int())
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		y := reflect.Indirect(bval)
// 		if x.Float() >= y.Float() {
// 			return float32(x.Float())
// 		}
// 		return float32(y.Float())
// 	case float32:
// 		if aval.Float() >= bval.Float() {
// 			return float32(aval.Float())
// 		}
// 		return float32(bval.Float())
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		y := reflect.Indirect(bval)
// 		if x.Float() >= y.Float() {
// 			return x.Float()
// 		}
// 		return y.Float()
// 	case float64:
// 		if aval.Float() >= bval.Float() {
// 			return aval.Float()
// 		}
// 		return bval.Float()
// 	default:
// 		log.Panic("maxint: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func maxf32(a, b float32) float32 {
	if a >= b {
		return a
	}
	return b
}

func maxf64(a, b float64) float64 {
	if a >= b {
		return a
	}
	return b
}

func maxint(a, b int) int {
	if a >= b {
		return a
	}
	return b
}

// func minint(a, b interface{}) interface{} {
// 	aval := reflect.ValueOf(a)
// 	bval := reflect.ValueOf(b)

// 	if reflect.TypeOf(a) != reflect.TypeOf(b) {
// 		log.Panic("minint: mismatched data types: ", reflect.TypeOf(a), reflect.TypeOf(b))
// 	}

// 	switch a.(_type) {
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		y := reflect.Indirect(bval)
// 		if x.Int() <= y.Int() {
// 			return int(x.Int())
// 		}
// 		return int(y.Int())
// 	case int:
// 		if aval.Int() <= bval.Int() {
// 			return int(aval.Int())
// 		}
// 		return int(bval.Int())
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		y := reflect.Indirect(bval)
// 		if x.Float() <= y.Float() {
// 			return float32(x.Float())
// 		}
// 		return float32(y.Float())
// 	case float32:
// 		if aval.Float() <= bval.Float() {
// 			return float32(aval.Float())
// 		}
// 		return float32(bval.Float())
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		y := reflect.Indirect(bval)
// 		if x.Float() <= y.Float() {
// 			return x.Float()
// 		}
// 		return y.Float()
// 	case float64:
// 		if aval.Float() <= bval.Float() {
// 			return aval.Float()
// 		}
// 		return bval.Float()
// 	default:
// 		log.Panic("minint: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func minf32(a, b float32) float32 {
	if a <= b {
		return a
	}
	return b
}

func minf64(a, b float64) float64 {
	if a <= b {
		return a
	}
	return b
}

func minint(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// func mod(a, b interface{}) interface{} {
// 	aval := reflect.ValueOf(a)
// 	bval := reflect.ValueOf(b)

// 	if reflect.TypeOf(a) != reflect.TypeOf(b) {
// 		log.Panic("mod: mismatched data types: ", reflect.TypeOf(a), reflect.TypeOf(b))
// 	}

// 	switch a.(_type) {
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		y := reflect.Indirect(bval)
// 		return int(x.Int() % y.Int())
// 	case int:
// 		return int(aval.Int() % bval.Int())
// 	default:
// 		log.Panic("mod: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func modint(a, b int) int {
	return a % b
}

// func nint(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		return int(math.Round(float64(x.Int())))
// 	case int:
// 		return int(math.Round(float64(aval.Int())))
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return int(math.Round(x.Float()))
// 	case float32:
// 		return int(math.Round(aval.Float()))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return int(math.Round(x.Float()))
// 	case float64:
// 		return int(math.Round(aval.Float()))
// 	default:
// 		log.Panic("nint: unrecognized parameter _type a: ", reflect.TypeOf(a))
// 	}

// 	return nil
// }

func nintf32(a float32) int {
	return int(math.Round(float64(a)))
}

// func power(a, b interface{}) interface{} {
// 	var bout float64

// 	aval := reflect.ValueOf(a)
// 	bval := reflect.ValueOf(b)

// 	switch b.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(bval)
// 		bout = x.Float()
// 	case float32:
// 		bout = bval.Float()
// 	case *float64:
// 		x := reflect.Indirect(bval)
// 		bout = x.Float()
// 	case float64:
// 		bout = bval.Float()
// 	case *int:
// 		x := reflect.Indirect(bval)
// 		bout = float64(x.Int())
// 	case int:
// 		bout = float64(bval.Int())
// 	default:
// 		log.Panic("power: unrecognized parameter _type b: ", reflect.TypeOf(a))
// 	}

// 	switch a.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return float32(math.Pow(x.Float(), bout))
// 	case float32:
// 		return float32(math.Pow(aval.Float(), bout))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return math.Pow(x.Float(), bout)
// 	case float64:
// 		return math.Pow(aval.Float(), bout)
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		return int(math.Pow(float64(x.Int()), bout))
// 	case int:
// 		return int(math.Pow(float64(aval.Int()), bout))
// 	default:
// 		log.Panic("power: unrecognized parameter _type a: ", reflect.TypeOf(a))
// 	}

// 	return nil
// }

func powf32(a, b float32) float32 {
	return float32(math.Pow(float64(a), float64(b)))
}

func powf64(a, b float64) float64 {
	return math.Pow(a, b)
}

func powint(a, b int) int {
	return int(math.Pow(float64(a), float64(b)))
}

// func re(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return float32(real(x.Complex()))
// 	case complex64:
// 		return float32(real(aval.Complex()))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return real(x.Complex())
// 	case complex128:
// 		return real(aval.Complex())
// 	default:
// 		log.Panic("re: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func roundf32(a float32) float32 {
	var sd int = 8 // significant digits of a float32
	sc := float32(math.Pow10(sd))
	b := int(a * sc)
	return float32(b) / sc
}

func roundf64(a float64) float64 {
	var sd int = 16 // significant digits of a float64
	sc := math.Pow10(sd)
	b := int(a * sc)
	return float64(b) / sc
}

// func sign(a, b interface{}) interface{} {
// 	var bout float64

// 	aval := reflect.ValueOf(a)
// 	bval := reflect.ValueOf(b)

// 	switch b.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(bval)
// 		bout = x.Float()
// 	case float32:
// 		bout = bval.Float()
// 	case *float64:
// 		x := reflect.Indirect(bval)
// 		bout = x.Float()
// 	case float64:
// 		bout = bval.Float()
// 	case *int:
// 		x := reflect.Indirect(bval)
// 		bout = float64(x.Int())
// 	case int:
// 		bout = float64(bval.Int())
// 	default:
// 		log.Panic("sign: unrecognized parameter _type b: ", reflect.TypeOf(a))
// 	}

// 	switch b.(_type) {
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		return int(math.Copysign(float64(x.Int()), bout))
// 	case int:
// 		return int(math.Copysign(float64(aval.Int()), bout))
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return float32(math.Copysign(x.Float(), bout))
// 	case float32:
// 		return float32(math.Copysign(aval.Float(), bout))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return math.Copysign(x.Float(), bout)
// 	case float64:
// 		return math.Copysign(aval.Float(), bout)
// 	default:
// 		log.Panic("sign: unrecognized parameter _type a: ", reflect.TypeOf(a))
// 	}

// 	return nil
// }

func signf32(a, b float32) float32 {
	return float32(math.Copysign(float64(a), float64(b)))
}

func signf64(a, b float64) float64 {
	return math.Copysign(a, b)
}

func signint(a, b int) int {
	return int(math.Copysign(float64(a), float64(b)))
}

// func sin(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return float32(math.Sin(x.Float()))
// 	case float32:
// 		return float32(math.Sin(aval.Float()))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return math.Sin(x.Float())
// 	case float64:
// 		return math.Sin(aval.Float())
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return complex64(cmplx.Sin(x.Complex()))
// 	case complex64:
// 		return complex64(cmplx.Sin(aval.Complex()))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return cmplx.Sin(x.Complex())
// 	case complex128:
// 		return cmplx.Sin(aval.Complex())
// 	case *int:
// 		x := reflect.Indirect(aval)
// 		return int(math.Sin(float64(x.Int())))
// 	case int:
// 		return int(math.Sin(float64(aval.Int())))
// 	default:
// 		log.Panic("sin: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func sinf32(a float32) float32 {
	return float32(math.Sin(float64(a)))
}

// func sqrt(a interface{}) interface{} {
// 	aval := reflect.ValueOf(a)

// 	switch a.(_type) {
// 	case *float32:
// 		x := reflect.Indirect(aval)
// 		return float32(math.Sqrt(x.Float()))
// 	case float32:
// 		return float32(math.Sqrt(aval.Float()))
// 	case *float64:
// 		x := reflect.Indirect(aval)
// 		return math.Sqrt(x.Float())
// 	case float64:
// 		return math.Sqrt(aval.Float())
// 	case *complex64:
// 		x := reflect.Indirect(aval)
// 		return complex64(cmplx.Sqrt(x.Complex()))
// 	case complex64:
// 		return complex64(cmplx.Sqrt(aval.Complex()))
// 	case *complex128:
// 		x := reflect.Indirect(aval)
// 		return cmplx.Sqrt(x.Complex())
// 	case complex128:
// 		return cmplx.Sqrt(aval.Complex())
// 	default:
// 		log.Panic("sqrt: unrecognized parameter _type: ", reflect.TypeOf(a))
// 	}
// 	return nil
// }

func sqrtf32(a float32) float32 {
	return float32(math.Sqrt(float64(a)))
}

func sqrtf64(a float64) float64 {
	return math.Sqrt(a)
}
