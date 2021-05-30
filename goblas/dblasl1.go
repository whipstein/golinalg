package goblas

import (
	"golinalg/mat"
	"math"
)

// Dasum takes the sum of the absolute values
func Dasum(n *int, dx *mat.Vector, incx *int) (dasumReturn float64) {
	var i, m, mp1, nincx int

	if (*n) <= 0 || (*incx) <= 0 {
		return
	}
	if (*incx) == 1 {
		//        code for increment equal to 1
		m = (*n) % 6
		if m != 0 {
			for i = 1; i <= m; i++ {
				dasumReturn += math.Abs(dx.Get(i - 1))
			}
			if (*n) < 6 {
				return
			}
		}
		mp1 = m + 1
		for i = mp1; i <= (*n); i += 6 {
			dasumReturn += math.Abs(dx.Get(i-1)) + math.Abs(dx.Get(i+1-1)) + math.Abs(dx.Get(i+2-1)) + math.Abs(dx.Get(i+3-1)) + math.Abs(dx.Get(i+4-1)) + math.Abs(dx.Get(i+5-1))
		}
	} else {
		//
		//        code for increment not equal to 1
		//
		nincx = (*n) * (*incx)
		for i = 1; i <= nincx; i += (*incx) {
			dasumReturn += math.Abs(dx.Get(i - 1))
		}
	}
	return
}

// Daxpy constant times a vector plus a vector.
func Daxpy(n *int, da *float64, dx *mat.Vector, incx *int, dy *mat.Vector, incy *int) {
	var i, ix, iy, m, mp1 int

	if (*n) <= 0 {
		return
	}
	if (*da) == 0.0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//
		//        code for both increments equal to 1
		//
		//
		//        clean-up loop
		//
		m = (*n) % 4
		if m != 0 {
			for i = 1; i <= m; i++ {
				dy.Set(i-1, dy.Get(i-1)+(*da)*dx.Get(i-1))
			}
		}
		if (*n) < 4 {
			return
		}
		mp1 = m + 1
		for i = mp1; i <= (*n); i += 4 {
			dy.Set(i-1, dy.Get(i-1)+(*da)*dx.Get(i-1))
			dy.Set(i+1-1, dy.Get(i+1-1)+(*da)*dx.Get(i+1-1))
			dy.Set(i+2-1, dy.Get(i+2-1)+(*da)*dx.Get(i+2-1))
			dy.Set(i+3-1, dy.Get(i+3-1)+(*da)*dx.Get(i+3-1))
		}
	} else {
		//
		//        code for unequal increments or equal increments
		//          not equal to 1
		//
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			dy.Set(iy-1, dy.Get(iy-1)+(*da)*dx.Get(ix-1))
			ix += (*incx)
			iy += (*incy)
		}
	}
	return
}

// Dcopy copies a vector, x, to a vector, y.
func Dcopy(n *int, dx *mat.Vector, incx *int, dy *mat.Vector, incy *int) {
	var i, ix, iy, m, mp1 int

	if (*n) <= 0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//
		//        code for both increments equal to 1
		//
		//
		//        clean-up loop
		//
		m = (*n) % 7
		if m != 0 {
			for i = 1; i <= m; i++ {
				dy.Set(i-1, dx.Get(i-1))
			}
			if (*n) < 7 {
				return
			}
		}
		mp1 = m + 1
		for i = mp1; i <= (*n); i += 7 {
			dy.Set(i-1, dx.Get(i-1))
			dy.Set(i+1-1, dx.Get(i+1-1))
			dy.Set(i+2-1, dx.Get(i+2-1))
			dy.Set(i+3-1, dx.Get(i+3-1))
			dy.Set(i+4-1, dx.Get(i+4-1))
			dy.Set(i+5-1, dx.Get(i+5-1))
			dy.Set(i+6-1, dx.Get(i+6-1))
		}
	} else {
		//
		//        code for unequal increments or equal increments
		//          not equal to 1
		//
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			dy.Set(iy-1, dx.Get(ix-1))
			ix += (*incx)
			iy += (*incy)
		}
	}
	return
}

// Ddot forms the dot product of two vectors.
func Ddot(n *int, dx *mat.Vector, incx *int, dy *mat.Vector, incy *int) (ddotReturn float64) {
	var i, ix, iy, m, mp1 int

	if (*n) <= 0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//
		//        code for both increments equal to 1
		//
		//
		//        clean-up loop
		//
		m = (*n) % 5
		if m != 0 {
			for i = 1; i <= m; i++ {
				ddotReturn += dx.Get(i-1) * dy.Get(i-1)
			}
			if (*n) < 5 {
				return
			}
		}
		mp1 = m + 1
		for i = mp1; i <= (*n); i += 5 {
			ddotReturn += dx.Get(i-1)*dy.Get(i-1) + dx.Get(i+1-1)*dy.Get(i+1-1) + dx.Get(i+2-1)*dy.Get(i+2-1) + dx.Get(i+3-1)*dy.Get(i+3-1) + dx.Get(i+4-1)*dy.Get(i+4-1)
		}
	} else {
		//
		//        code for unequal increments or equal increments
		//          not equal to 1
		//
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			ddotReturn += dx.Get(ix-1) * dy.Get(iy-1)
			ix += (*incx)
			iy += (*incy)
		}
	}
	return
}

// Dnrm2 returns the euclidean norm of a vector via the function
// name, so that Dnrm2 := sqrt( x'*x )
func Dnrm2(n *int, x *mat.Vector, incx *int) (dnrm2Return float64) {
	var absxi, one, scale, ssq, zero float64
	var ix int

	one = 1.0
	zero = 0.0

	if (*n) < 1 || (*incx) < 1 {
		dnrm2Return = zero
	} else if (*n) == 1 {
		dnrm2Return = math.Abs(x.Get(0))
	} else {
		scale = zero
		ssq = one
		//        The following loop is equivalent to this call to the LAPACK
		//        auxiliary routine:
		//        CALL DLASSQ( N, X, INCX, SCALE, SSQ )
		//
		for ix = 1; ix <= 1+((*n)-1)*(*incx); ix += (*incx) {
			if x.Get(ix-1) != zero {
				absxi = math.Abs(x.Get(ix - 1))
				if scale < absxi {
					ssq = one + ssq*math.Pow(scale/absxi, 2)
					scale = absxi
				} else {
					ssq += math.Pow(absxi/scale, 2)
				}
			}
		}
		dnrm2Return = scale * math.Sqrt(ssq)
	}

	return
}

// Drot applies a plane rotation
func Drot(n *int, dx *mat.Vector, incx *int, dy *mat.Vector, incy *int, c, s *float64) {
	var dtemp float64
	var i, ix, iy int

	if (*n) <= 0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//
		//       code for both increments equal to 1
		//
		for i = 1; i <= (*n); i++ {
			dtemp = (*c)*dx.Get(i-1) + (*s)*dy.Get(i-1)
			dy.Set(i-1, (*c)*dy.Get(i-1)-(*s)*dx.Get(i-1))
			dx.Set(i-1, dtemp)
		}
	} else {
		//
		//       code for unequal increments or equal increments not equal
		//         to 1
		//
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			dtemp = (*c)*dx.Get(ix-1) + (*s)*dy.Get(iy-1)
			dy.Set(iy-1, (*c)*dy.Get(iy-1)-(*s)*dx.Get(ix-1))
			dx.Set(ix-1, dtemp)
			ix += (*incx)
			iy += (*incy)
		}
	}
}

// Drotg construct givens plane rotation
func Drotg(da, db, c, s *float64) {
	var r, roe, scale, z float64

	roe = *db
	if math.Abs(*da) > math.Abs(*db) {
		roe = *da
	}
	scale = math.Abs(*da) + math.Abs(*db)
	if scale == 0.0 {
		(*c) = 1.0
		(*s) = 0.0
		r = 0.0
		z = 0.0
	} else {
		r = scale * math.Sqrt(math.Pow((*da)/scale, 2)+math.Pow((*db)/scale, 2))
		r = math.Copysign(1, roe) * r
		*c = (*da) / r
		*s = (*db) / r
		z = 1.0
		if math.Abs(*da) > math.Abs(*db) {
			z = (*s)
		}
		if math.Abs(*db) >= math.Abs(*da) && (*c) != 0.0 {
			z = 1.0 / (*c)
		}
	}
	*da = r
	*db = z
}

// Drotm applies the modified Givens transformation, H, to the 2 x n matrix
//    (DX**T) , where **T indicates transpose. The elements of dx are in
//    (DY**T)
//
//    DX(LX+I*INCX), I = 0 TO N-1, WHERE LX = 1 IF INCX .GE. 0, ELSE
//    LX = (-INCX)*N, AND SIMILARLY FOR SY USING LY AND INCY.
//    WITH DPARAM(1)=DFLAG, H HAS ONE OF THE FOLLOWING FORMS..
//
//    DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0
//
//      (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
//    H=(          )    (          )    (          )    (          )
//      (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
//    SEE DROTMG FOR A DESCRIPTION OF DATA STORAGE IN DPARAM.
func Drotm(n *int, dx *mat.Vector, incx *int, dy *mat.Vector, incy *int, dparam *mat.DrotMatrix) {
	var dh11, dh12, dh21, dh22, w, z float64
	var dflag, i, kx, ky, nsteps int

	dflag = dparam.Flag
	if (*n) <= 0 || (dflag+2 == 0) {
		return
	}
	if (*incx) == (*incy) && (*incx) > 0 {

		nsteps = (*n) * (*incx)
		if dflag < 0 {
			dh11 = dparam.H11
			dh12 = dparam.H12
			dh21 = dparam.H21
			dh22 = dparam.H22
			for i = 1; i <= nsteps; i += (*incx) {
				w = dx.Get(i - 1)
				z = dy.Get(i - 1)
				dx.Set(i-1, w*dh11+z*dh12)
				dy.Set(i-1, w*dh21+z*dh22)
			}
		} else if dflag == 0 {
			dh12 = dparam.H12
			dh21 = dparam.H21
			for i = 1; i <= nsteps; i += (*incx) {
				w = dx.Get(i - 1)
				z = dy.Get(i - 1)
				dx.Set(i-1, w+z*dh12)
				dy.Set(i-1, w*dh21+z)
			}
		} else {
			dh11 = dparam.H11
			dh22 = dparam.H22
			for i = 1; i <= nsteps; i += (*incx) {
				w = dx.Get(i - 1)
				z = dy.Get(i - 1)
				dx.Set(i-1, w*dh11+z)
				dy.Set(i-1, -w+dh22*z)
			}
		}
	} else {
		kx = 1
		ky = 1
		if (*incx) < 0 {
			kx = 1 + (1-(*n))*(*incx)
		}
		if (*incy) < 0 {
			ky = 1 + (1-(*n))*(*incy)
		}
		//
		if dflag < 0 {
			dh11 = dparam.H11
			dh12 = dparam.H12
			dh21 = dparam.H21
			dh22 = dparam.H22
			for i = 1; i <= (*n); i++ {
				w = dx.Get(kx - 1)
				z = dy.Get(ky - 1)
				dx.Set(kx-1, w*dh11+z*dh12)
				dy.Set(ky-1, w*dh21+z*dh22)
				kx += (*incx)
				ky += (*incy)
			}
		} else if dflag == 0 {
			dh12 = dparam.H12
			dh21 = dparam.H21
			for i = 1; i <= (*n); i++ {
				w = dx.Get(kx - 1)
				z = dy.Get(ky - 1)
				dx.Set(kx-1, w+z*dh12)
				dy.Set(ky-1, w*dh21+z)
				kx += (*incx)
				ky += (*incy)
			}
		} else {
			dh11 = dparam.H11
			dh22 = dparam.H22
			for i = 1; i <= (*n); i++ {
				w = dx.Get(kx - 1)
				z = dy.Get(ky - 1)
				dx.Set(kx-1, w*dh11+z)
				dy.Set(ky-1, -w+dh22*z)
				kx += (*incx)
				ky += (*incy)
			}
		}
	}
	return
}

// Drotmg constructs the modified Givens transformation matrix H which zeros
//    the second component of the 2-vector  (DSQRT(DD1)*DX1,DSQRT(DD2)*>    DY2)**T.
//    WITH DPARAM(1)=DFLAG, H HAS ONE OF THE FOLLOWING FORMS..
//
//    DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0
//
//      (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
//    H=(          )    (          )    (          )    (          )
//      (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
//    LOCATIONS 2-4 OF DPARAM CONTAIN DH11, DH21, DH12, AND DH22
//    RESPECTIVELY. (VALUES OF 1.D0, -1.D0, OR 0.D0 IMPLIED BY THE
//    VALUE OF DPARAM(1) ARE NOT STORED IN DPARAM.)
//
//    THE VALUES OF GAMSQ AND RGAMSQ SET IN THE DATA STATEMENT MAY BE
//    INEXACT.  THIS IS OK AS THEY ARE ONLY USED FOR TESTING THE SIZE
//    OF DD1 AND DD2.  ALL ACTUAL SCALING OF DATA IS DONE USING GAM.
func Drotmg(dd1, dd2, dx1, dy1 *float64, dparam *mat.DrotMatrix) {
	var dh11, dh12, dh21, dh22, dp1, dp2, dq1, dq2, dtemp, du, gam, gamsq, one, rgamsq, zero float64
	var dflag int
	dmb := mat.CreateDrotMatrixBuilder(dparam)

	zero, one = 0., 1.
	gam, gamsq, rgamsq = 4096., 16777216., 5.9604645e-8

	if *dd1 < zero {
		//        GO ZERO-H-D-AND-DX1..
		dflag = -1
		dh11 = zero
		dh12 = zero
		dh21 = zero
		dh22 = zero

		*dd1 = zero
		*dd2 = zero
		*dx1 = zero
	} else {
		//        CASE-DD1-NONNEGATIVE
		dp2 = (*dd2) * (*dy1)
		if dp2 == zero {
			dflag = -2
			dparam = dmb.Flag(dflag).H([4]float64{dh11, dh21, dh12, dh22}).Build()
			return
		}
		//        REGULAR-CASE..
		dp1 = (*dd1) * (*dx1)
		dq2 = dp2 * (*dy1)
		dq1 = dp1 * (*dx1)
		//
		if math.Abs(dq1) > math.Abs(dq2) {
			dh21 = -(*dy1) / (*dx1)
			dh12 = dp2 / dp1
			//
			du = one - dh12*dh21
			//
			if du > zero {
				dflag = 0
				*dd1 /= du
				*dd2 /= du
				*dx1 *= du
			}
		} else {
			if dq2 < zero {
				//              GO ZERO-H-D-AND-DX1..
				dflag = -1
				dh11 = zero
				dh12 = zero
				dh21 = zero
				dh22 = zero

				*dd1 = zero
				*dd2 = zero
				*dx1 = zero
			} else {
				dflag = 1
				dh11 = dp1 / dp2
				dh22 = (*dx1) / (*dy1)
				du = one + dh11*dh22
				dtemp = (*dd2) / du
				*dd2 = (*dd1) / du
				*dd1 = dtemp
				*dx1 = (*dy1) * du
			}
		}
		//     PROCEDURE..SCALE-CHECK
		if *dd1 != zero {
			for (*dd1 <= rgamsq) || (*dd1 >= gamsq) {
				if dflag == 0 {
					dh11 = one
					dh22 = one
					dflag = -1
				} else {
					dh21 = -one
					dh12 = one
					dflag = -1
				}
				if *dd1 <= rgamsq {
					*dd1 *= math.Pow(gam, 2)
					*dx1 /= gam
					dh11 /= gam
					dh12 /= gam
				} else {
					*dd1 /= math.Pow(gam, 2)
					*dx1 *= gam
					dh11 *= gam
					dh12 *= gam
				}
			}
		}
		if *dd2 != zero {
			for (math.Abs(*dd2) <= rgamsq) || (math.Abs(*dd2) >= gamsq) {
				if dflag == 0 {
					dh11 = one
					dh22 = one
					dflag = -1
				} else {
					dh21 = -one
					dh12 = one
					dflag = -1
				}
				if math.Abs(*dd2) <= rgamsq {
					*dd2 *= math.Pow(gam, 2)
					dh21 /= gam
					dh22 /= gam
				} else {
					*dd2 /= math.Pow(gam, 2)
					dh21 *= gam
					dh22 *= gam
				}
			}
		}
	}
	if dflag < 0 {
		dmb.Flag(dflag).H([4]float64{dh11, dh21, dh12, dh22}).Build()
	} else if dflag == 0 {
		dmb.Flag(dflag).H([4]float64{0, dh21, dh12, 0}).Build()
	} else {
		dmb.Flag(dflag).H([4]float64{dh11, 0, 0, dh22}).Build()
	}
	return
}

// Dscal scales a vector by a constant
func Dscal(n *int, da *float64, dx *mat.Vector, incx *int) {
	var i, m, mp1, nincx int

	if (*n) <= 0 || (*incx) <= 0 {
		return
	}
	if (*incx) == 1 {
		//
		//        code for increment equal to 1
		//
		//
		//        clean-up loop
		//
		m = (*n) % 5
		if m != 0 {
			for i = 1; i <= m; i++ {
				dx.Set(i-1, (*da)*dx.Get(i-1))
			}
			if (*n) < 5 {
				return
			}
		}
		mp1 = m + 1
		for i = mp1; i <= (*n); i += 5 {
			dx.Set(i-1, (*da)*dx.Get(i-1))
			dx.Set(i+1-1, (*da)*dx.Get(i+1-1))
			dx.Set(i+2-1, (*da)*dx.Get(i+2-1))
			dx.Set(i+3-1, (*da)*dx.Get(i+3-1))
			dx.Set(i+4-1, (*da)*dx.Get(i+4-1))
		}
	} else {
		//
		//        code for increment not equal to 1
		//
		nincx = (*n) * (*incx)
		for i = 1; i <= nincx; i += (*incx) {
			dx.Set(i-1, (*da)*dx.Get(i-1))
		}
	}
	return
}

// Dswap interchanges two vectors
func Dswap(n *int, dx *mat.Vector, incx *int, dy *mat.Vector, incy *int) {
	var dtemp float64
	var i, ix, iy, m, mp1 int

	if (*n) <= 0 {
		return
	}
	if (*incx) == 1 && (*incy) == 1 {
		//
		//       code for both increments equal to 1
		//
		//
		//       clean-up loop
		//
		m = (*n) % 3
		if m != 0 {
			for i = 1; i <= m; i++ {
				dtemp = dx.Get(i - 1)
				dx.Set(i-1, dy.Get(i-1))
				dy.Set(i-1, dtemp)
			}
			if (*n) < 3 {
				return
			}
		}
		mp1 = m + 1
		for i = mp1; i <= (*n); i += 3 {
			dtemp = dx.Get(i - 1)
			dx.Set(i-1, dy.Get(i-1))
			dy.Set(i-1, dtemp)
			dtemp = dx.Get(i + 1 - 1)
			dx.Set(i+1-1, dy.Get(i+1-1))
			dy.Set(i+1-1, dtemp)
			dtemp = dx.Get(i + 2 - 1)
			dx.Set(i+2-1, dy.Get(i+2-1))
			dy.Set(i+2-1, dtemp)
		}
	} else {
		//
		//       code for unequal increments or equal increments not equal
		//         to 1
		//
		ix = 1
		iy = 1
		if (*incx) < 0 {
			ix = (-(*n)+1)*(*incx) + 1
		}
		if (*incy) < 0 {
			iy = (-(*n)+1)*(*incy) + 1
		}
		for i = 1; i <= (*n); i++ {
			dtemp = dx.Get(ix - 1)
			dx.Set(ix-1, dy.Get(iy-1))
			dy.Set(iy-1, dtemp)
			ix += (*incx)
			iy += (*incy)
		}
	}
	return
}

// Idamax finds the index of the first element having maximum absolute value
func Idamax(n *int, dx *mat.Vector, incx *int) (idamaxReturn int) {
	var dmax float64
	var i, ix int

	if (*n) < 1 || (*incx) <= 0 {
		return
	}
	idamaxReturn = 1
	if (*n) == 1 {
		return
	}
	if (*incx) == 1 {
		//
		//        code for increment equal to 1
		//
		dmax = math.Abs(dx.Get(0))
		for i = 2; i <= (*n); i++ {
			if math.Abs(dx.Get(i-1)) > dmax {
				idamaxReturn = i
				dmax = math.Abs(dx.Get(i - 1))
			}
		}
	} else {
		//
		//        code for increment not equal to 1
		//
		ix = 1
		dmax = math.Abs(dx.Get(0))
		ix += (*incx)
		for i = 2; i <= (*n); i++ {
			if math.Abs(dx.Get(ix-1)) > dmax {
				idamaxReturn = i
				dmax = math.Abs(dx.Get(ix - 1))
			}
			ix += (*incx)
		}
	}
	return
}
