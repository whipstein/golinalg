package goblas

import (
	"golinalg/mat"
	"math"
	"math/cmplx"
)

// Dcabs1 computes |Re(.)| + |Im(.)| of a double complex number
func dcabs1(z complex128) (dcabs1Return float64) {

	dcabs1Return = math.Abs(real(z)) + math.Abs(imag(z))
	return
}

// Dzasum takes the sum of the (|Re(.)| + |Im(.)|)'s of a complex vector and
//    returns a single precision result.
func Dzasum(n *int, zx *mat.CVector, incx *int) (dzasumReturn float64) {
	var stemp float64
	var i, nincx int

	dzasumReturn = 0.0
	stemp = 0.0
	if (*n) <= 0 || (*incx) <= 0 {
		return
	}
	if (*incx) == 1 {
		//        code for increment equal to 1
		for i = 1; i <= (*n); i++ {
			stemp = stemp + dcabs1(zx.Get(i-1))
		}
	} else {
		//        code for increment not equal to 1
		nincx = (*n) * (*incx)
		for i = 1; i <= nincx; i += (*incx) {
			stemp = stemp + dcabs1(zx.Get(i-1))
		}
	}
	dzasumReturn = stemp
	return
}

// Dznrm2 returns the euclidean norm of a vector via the function
// name, so that
//
//    DZNRM2 := sqrt( x**H*x )
func Dznrm2(n *int, x *mat.CVector, incx *int) (dznrm2Return float64) {
	var norm, one, scale, ssq, temp, zero float64
	var ix int

	one = 1.0
	zero = 0.0

	if (*n) < 1 || (*incx) < 1 {
		norm = zero
	} else {
		scale = zero
		ssq = one
		//        The following loop is equivalent to this call to the LAPACK
		//        auxiliary routine:
		//        CALL ZLASSQ( N, X, INCX, SCALE, SSQ )
		for ix = 1; ix <= 1+((*n)-1)*(*incx); ix += (*incx) {
			if real(x.Get(ix-1)) != zero {
				temp = cmplx.Abs(x.GetReCmplx(ix - 1))
				if scale < temp {
					ssq = one + ssq*math.Pow(scale/temp, 2)
					scale = temp
				} else {
					ssq = ssq + math.Pow(temp/scale, 2)
				}
			}
			if imag(x.Get(ix-1)) != zero {
				temp = cmplx.Abs(x.GetImCmplx(ix - 1))
				if scale < temp {
					ssq = one + ssq*math.Pow(scale/temp, 2)
					scale = temp
				} else {
					ssq = ssq + math.Pow(temp/scale, 2)
				}
			}
		}
		norm = scale * math.Sqrt(ssq)
	}

	dznrm2Return = norm
	return
}

// Izamax finds the index of the first element having maximum |Re(.)| + |Im(.)|
func Izamax(n *int, zx *mat.CVector, incx *int) (izamaxReturn int) {
	var dmax float64
	var i, ix int

	izamaxReturn = 0
	if (*n) < 1 || (*incx) <= 0 {
		return
	}
	izamaxReturn = 1
	if (*n) == 1 {
		return
	}
	if (*incx) == 1 {
		//        code for increment equal to 1
		dmax = dcabs1(zx.Get(0))
		for i = 2; i <= (*n); i++ {
			if dcabs1(zx.Get(i-1)) > dmax {
				izamaxReturn = i
				dmax = dcabs1(zx.Get(i - 1))
			}
		}
	} else {
		//        code for increment not equal to 1
		ix = 1
		dmax = dcabs1(zx.Get(0))
		ix = ix + (*incx)
		for i = 2; i <= (*n); i++ {
			if dcabs1(zx.Get(ix-1)) > dmax {
				izamaxReturn = i
				dmax = dcabs1(zx.Get(ix - 1))
			}
			ix = ix + (*incx)
		}
	}
	return
}

// Zdscal scales a vector by a constant.
func Zdscal(n *int, da *float64, zx *mat.CVector, incx *int) {
	var i, nincx int

	if (*n) <= 0 || (*incx) <= 0 {
		return
	}
	if (*incx) == 1 {
		//        code for increment equal to 1
		for i = 1; i <= (*n); i++ {
			zx.Set(i-1, complex(*da, 0)*zx.Get(i-1))
		}
	} else {
		//        code for increment not equal to 1
		nincx = (*n) * (*incx)
		for i = 1; i <= nincx; i += (*incx) {
			zx.Set(i-1, complex(*da, 0)*zx.Get(i-1))
		}
	}
	return
}

// Zscal scales a vector by a constant.
func Zscal(n *int, za *complex128, zx *mat.CVector, incx *int) {
	var i, nincx int

	if (*n) <= 0 || (*incx) <= 0 {
		return
	}
	if (*incx) == 1 {
		//        code for increment equal to 1
		for i = 1; i <= (*n); i++ {
			zx.Set(i-1, (*za)*zx.Get(i-1))
		}
	} else {
		//        code for increment not equal to 1
		nincx = (*n) * (*incx)
		for i = 1; i <= nincx; i += (*incx) {
			zx.Set(i-1, (*za)*zx.Get(i-1))
		}
	}
	return
}

// Zaxpy constant times a vector plus a vector.
func Zaxpy(n *int, za *complex128, zx *mat.CVector, incx *int, zy *mat.CVector, incy *int) {
	var i, ix, iy int

	if (*n) <= 0 {
		return
	}
	if dcabs1(*za) == 0.0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//        code for both increments equal to 1
		for i = 1; i <= (*n); i++ {
			zy.Set(i-1, zy.Get(i-1)+(*za)*zx.Get(i-1))
		}
	} else {
		//        code for unequal increments or equal increments
		//          not equal to 1
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			zy.Set(iy-1, zy.Get(iy-1)+(*za)*zx.Get(ix-1))
			ix = ix + (*incx)
			iy = iy + (*incy)
		}
	}
}

// Zcopy copies a vector, x, to a vector, y.
func Zcopy(n *int, zx *mat.CVector, incx *int, zy *mat.CVector, incy *int) {
	var i, ix, iy int

	if (*n) <= 0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//        code for both increments equal to 1
		for i = 1; i <= (*n); i++ {
			zy.Set(i-1, zx.Get(i-1))
		}
	} else {
		//        code for unequal increments or equal increments
		//          not equal to 1
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			zy.Set(iy-1, zx.Get(ix-1))
			ix = ix + (*incx)
			iy = iy + (*incy)
		}
	}
	return
}

// Zdotc forms the dot product of two complex vectors
//      ZDOTC = X^H * Y
func Zdotc(n *int, zx *mat.CVector, incx *int, zy *mat.CVector, incy *int) (zdotcReturn complex128) {
	var ztemp complex128
	var i, ix, iy int

	ztemp = (0.0 + 0.0*1i)
	zdotcReturn = (0.0 + 0.0*1i)
	if (*n) <= 0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//        code for both increments equal to 1
		for i = 1; i <= (*n); i++ {
			ztemp = ztemp + zx.GetConj(i-1)*zy.Get(i-1)
		}
	} else {
		//        code for unequal increments or equal increments
		//          not equal to 1
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			ztemp = ztemp + zx.GetConj(ix-1)*zy.Get(iy-1)
			ix = ix + (*incx)
			iy = iy + (*incy)
		}
	}
	zdotcReturn = ztemp
	return
}

// Zdotu forms the dot product of two complex vectors
//      ZDOTU = X^T * Y
func Zdotu(n *int, zx *mat.CVector, incx *int, zy *mat.CVector, incy *int) (zdotuReturn complex128) {
	var ztemp complex128
	var i, ix, iy int

	ztemp = (0.0 + 0.0*1i)
	zdotuReturn = (0.0 + 0.0*1i)
	if (*n) <= 0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//        code for both increments equal to 1
		for i = 1; i <= (*n); i++ {
			ztemp = ztemp + zx.Get(i-1)*zy.Get(i-1)
		}
	} else {
		//        code for unequal increments or equal increments
		//          not equal to 1
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			ztemp = ztemp + zx.Get(ix-1)*zy.Get(iy-1)
			ix = ix + (*incx)
			iy = iy + (*incy)
		}
	}
	zdotuReturn = ztemp
	return
}

// Zswap interchanges two vectors.
func Zswap(n *int, zx *mat.CVector, incx *int, zy *mat.CVector, incy *int) {
	var ztemp complex128
	var i, ix, iy int

	if (*n) <= 0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//       code for both increments equal to 1
		for i = 1; i <= (*n); i++ {
			ztemp = zx.Get(i - 1)
			zx.Set(i-1, zy.Get(i-1))
			zy.Set(i-1, ztemp)
		}
	} else {
		//       code for unequal increments or equal increments not equal
		//         to 1
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			ztemp = zx.Get(ix - 1)
			zx.Set(ix-1, zy.Get(iy-1))
			zy.Set(iy-1, ztemp)
			ix = ix + (*incx)
			iy = iy + (*incy)
		}
	}
}

// Zdrot Applies a plane rotation, where the cos and sin (c and s) are real
// and the vectors cx and cy are complex.
// jack dongarra, linpack, 3/11/78.
func Zdrot(n *int, cx *mat.CVector, incx *int, cy *mat.CVector, incy *int, c, s *float64) {
	var ctemp complex128
	var i, ix, iy int

	if (*n) <= 0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//        code for both increments equal to 1
		for i = 1; i <= (*n); i++ {
			ctemp = complex(*c, 0)*cx.Get(i-1) + complex(*s, 0)*cy.Get(i-1)
			cy.Set(i-1, complex(*c, 0)*cy.Get(i-1)-complex(*s, 0)*cx.Get(i-1))
			cx.Set(i-1, ctemp)
		}
	} else {
		//        code for unequal increments or equal increments not equal
		//          to 1
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			ctemp = complex(*c, 0)*cx.Get(ix-1) + complex(*s, 0)*cy.Get(iy-1)
			cy.Set(iy-1, complex(*c, 0)*cy.Get(iy-1)-complex(*s, 0)*cx.Get(ix-1))
			cx.Set(ix-1, ctemp)
			ix = ix + (*incx)
			iy = iy + (*incy)
		}
	}
}

// Zrotg determines a double complex Givens rotation.
func Zrotg(ca, cb *complex128, c *float64, s *complex128) {
	var alpha complex128
	var norm, scale float64

	if cmplx.Abs(*ca) == 0.0 {
		(*c) = 0.0
		(*s) = (1.0 + 0.0*1i)
		(*ca) = (*cb)
	} else {
		scale = cmplx.Abs(*ca) + cmplx.Abs(*cb)
		norm = scale * math.Sqrt(math.Pow(cmplx.Abs((*ca)/complex(scale, 0)), 2)+math.Pow(cmplx.Abs((*cb)/complex(scale, 0)), 2))
		alpha = (*ca) / complex(cmplx.Abs(*ca), 0)
		(*c) = cmplx.Abs(*ca) / norm
		(*s) = alpha * cmplx.Conj(*cb) / complex(norm, 0)
		(*ca) = alpha * complex(norm, 0)
	}
	return
}
