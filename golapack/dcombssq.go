package golapack

import (
	"golinalg/mat"
	"math"
)

// Dcombssq adds two scaled sum of squares quantities, V1 := V1 + V2.
// That is,
//
//    V1_scale**2 * V1_sumsq := V1_scale**2 * V1_sumsq
//                            + V2_scale**2 * V2_sumsq
func Dcombssq(v1, v2 *mat.Vector) {
	if v1.Get(0) >= v2.Get(0) {
		if v1.Get(0) != 0 {
			v1.Set(1, v1.Get(1)+math.Pow(v2.Get(0)/v1.Get(0), 2)*v2.Get(1))
		}
	} else {
		v1.Set(1, v2.Get(1)+math.Pow(v1.Get(0)/v2.Get(0), 2)*v1.Get(1))
		v1.Set(0, v2.Get(0))
	}
}
