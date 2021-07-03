package goblas

import (
	"math"
	"sync"

	"github.com/whipstein/golinalg/mat"
)

// Dasum takes the sum of the absolute values
func Dasum(n int, dx *mat.Vector, incx int) (dasumReturn float64) {
	if n <= 0 || incx <= 0 {
		return
	}

	for i := 0; i < n*incx; i += incx {
		dasumReturn += math.Abs(dx.Get(i))
	}
	return
}

// Daxpy constant times a vector plus a vector.
func Daxpy(n int, da float64, dx *mat.Vector, incx int, dy *mat.Vector, incy int) {
	var i, ix, iy int

	if n <= 0 || da == 0 {
		return
	}

	if incx < 0 {
		ix = (-n + 1) * incx
	}
	if incy < 0 {
		iy = (-n + 1) * incy
	}
	for i = 0; i < n; i, ix, iy = i+1, ix+incx, iy+incy {
		dy.Set(iy, dy.Get(iy)+da*dx.Get(ix))
	}
}

// Dcopy copies a vector, x, to a vector, y.
func Dcopy(n int, dx *mat.Vector, incx int, dy *mat.Vector, incy int) {
	var i, ix, iy int

	if n <= 0 {
		return
	}

	if incx < 0 {
		ix = (-n + 1) * incx
	}
	if incy < 0 {
		iy = (-n + 1) * incy
	}
	for i = 0; i < n; i, ix, iy = i+1, ix+incx, iy+incy {
		dy.Set(iy, dx.Get(ix))
	}
}

// Ddot forms the dot product of two vectors.
func Ddot(n int, dx *mat.Vector, incx int, dy *mat.Vector, incy int) (ddotReturn float64) {
	var i, ix, iy int

	if n <= 0 {
		return
	}

	if incx < 0 {
		ix = (-n + 1) * incx
	}
	if incy < 0 {
		iy = (-n + 1) * incy
	}
	for i = 0; i < n; i, ix, iy = i+1, ix+incx, iy+incy {
		ddotReturn += dx.Get(ix) * dy.Get(iy)
	}

	return
}

// Dnrm2 returns the euclidean norm of a vector via the function
// name, so that Dnrm2 := sqrt( x'*x )
func Dnrm2(n int, x *mat.Vector, incx int) (dnrm2Return float64) {
	var absxi, scale, ssq float64
	var ix int

	if n < 1 || incx < 1 {
		return
	} else if n == 1 {
		return math.Abs(x.Get(0))
	} else {
		ssq = 1
		//        The following loop is equivalent to this call to the LAPACK
		//        auxiliary routine:
		//        CALL DLASSQ( N, X, INCX, SCALE, SSQ )
		//
		for ix = 0; ix <= (n-1)*incx; ix += incx {
			if x.Get(ix) != 0 {
				absxi = math.Abs(x.Get(ix))
				if scale < absxi {
					ssq = 1 + ssq*math.Pow(scale/absxi, 2)
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
func Drot(n int, dx *mat.Vector, incx int, dy *mat.Vector, incy int, c, s float64) {
	if n <= 0 {
		return
	}

	blocksize := 512

	if n < minParBlocks*blocksize {
		drot(n, dx, incx, dy, incy, c, s)
	} else {
		nblocks := blocks(n, blocksize)
		var wg sync.WaitGroup
		defer wg.Wait()

		for i := 0; i < nblocks; i++ {
			size := blocksize
			if (i+1)*blocksize > n {
				size = n - i*blocksize
			}
			wg.Add(1)
			go func(i, size int) {
				defer wg.Done()
				drot(size, dx.Off(i*blocksize*incx), incx, dy.Off(i*blocksize*incy), incy, c, s)
			}(i, size)
		}
	}
}
func drot(n int, dx *mat.Vector, incx int, dy *mat.Vector, incy int, c, s float64) {
	var dtemp float64
	var i, ix, iy int

	if incx < 0 {
		ix = (-n + 1) * incx
	}
	if incy < 0 {
		iy = (-n + 1) * incy
	}
	for i = 0; i < n; i, ix, iy = i+1, ix+incx, iy+incy {
		dtemp = c*dx.Get(ix) + s*dy.Get(iy)
		dy.Set(iy, c*dy.Get(iy)-s*dx.Get(ix))
		dx.Set(ix, dtemp)
	}
}

// Drotg construct givens plane rotation
func Drotg(da, db, c, s float64) (daReturn, dbReturn, cReturn, sReturn float64) {
	var roe, scale float64

	roe = db
	if math.Abs(da) > math.Abs(db) {
		roe = da
	}
	scale = math.Abs(da) + math.Abs(db)
	if scale == 0.0 {
		cReturn = 1.0
		sReturn = 0.0
		daReturn = 0.0
		dbReturn = 0.0
	} else {
		daReturn = scale * math.Sqrt(math.Pow(da/scale, 2)+math.Pow(db/scale, 2))
		daReturn *= math.Copysign(1, roe)
		cReturn = da / daReturn
		sReturn = db / daReturn
		dbReturn = 1.0
		if math.Abs(da) > math.Abs(db) {
			dbReturn = sReturn
		}
		if math.Abs(db) >= math.Abs(da) && cReturn != 0.0 {
			dbReturn = 1.0 / cReturn
		}
	}

	return
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
func Drotm(n int, dx *mat.Vector, incx int, dy *mat.Vector, incy int, dparam *mat.DrotMatrix) {
	var dh11, dh12, dh21, dh22, w, z float64
	var dflag, i, kx, ky, nsteps int

	dflag = dparam.Flag
	if n <= 0 || (dflag+2 == 0) {
		return
	}
	if incx == incy && incx > 0 {

		nsteps = n * incx
		if dflag < 0 {
			dh11 = dparam.H11
			dh12 = dparam.H12
			dh21 = dparam.H21
			dh22 = dparam.H22
			for i = 0; i < nsteps; i += incx {
				w = dx.Get(i)
				z = dy.Get(i)
				dx.Set(i, w*dh11+z*dh12)
				dy.Set(i, w*dh21+z*dh22)
			}
		} else if dflag == 0 {
			dh12 = dparam.H12
			dh21 = dparam.H21
			for i = 0; i < nsteps; i += incx {
				w = dx.Get(i)
				z = dy.Get(i)
				dx.Set(i, w+z*dh12)
				dy.Set(i, w*dh21+z)
			}
		} else {
			dh11 = dparam.H11
			dh22 = dparam.H22
			for i = 0; i < nsteps; i += incx {
				w = dx.Get(i)
				z = dy.Get(i)
				dx.Set(i, w*dh11+z)
				dy.Set(i, -w+dh22*z)
			}
		}
	} else {
		if incx < 0 {
			kx = (1 - n) * incx
		}
		if incy < 0 {
			ky = (1 - n) * incy
		}

		if dflag < 0 {
			dh11 = dparam.H11
			dh12 = dparam.H12
			dh21 = dparam.H21
			dh22 = dparam.H22
			for i = 0; i < n; i, kx, ky = i+1, kx+incx, ky+incy {
				w = dx.Get(kx)
				z = dy.Get(ky)
				dx.Set(kx, w*dh11+z*dh12)
				dy.Set(ky, w*dh21+z*dh22)
			}
		} else if dflag == 0 {
			dh12 = dparam.H12
			dh21 = dparam.H21
			for i = 0; i < n; i, kx, ky = i+1, kx+incx, ky+incy {
				w = dx.Get(kx)
				z = dy.Get(ky)
				dx.Set(kx, w+z*dh12)
				dy.Set(ky, w*dh21+z)
			}
		} else {
			dh11 = dparam.H11
			dh22 = dparam.H22
			for i = 0; i < n; i, kx, ky = i+1, kx+incx, ky+incy {
				w = dx.Get(kx)
				z = dy.Get(ky)
				dx.Set(kx, w*dh11+z)
				dy.Set(ky, -w+dh22*z)
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
func Drotmg(dd1, dd2, dx1, dy1 float64) (dd1Return, dd2Return, dx1Return float64, dparamReturn *mat.DrotMatrix) {
	var dh11, dh12, dh21, dh22, dp1, dp2, dq1, dq2, du, gam, gamsq, rgamsq float64
	var dflag int
	dmb := mat.NewDrotMatrixBuilder()
	dd1Return, dd2Return, dx1Return = dd1, dd2, dx1

	gam, gamsq, rgamsq = 4096., 16777216., 5.9604645e-8

	if dd1Return < 0 {
		//        GO ZERO-H-D-AND-DX1..
		dflag = -1
		dh11 = 0
		dh12 = 0
		dh21 = 0
		dh22 = 0

		dd1Return = 0
		dd2Return = 0
		dx1Return = 0
	} else {
		//        CASE-DD1-NONNEGATIVE
		dp2 = dd2Return * dy1
		if dp2 == 0 {
			dflag = -2
			dparamReturn = dmb.Flag(dflag).H([4]float64{dh11, dh21, dh12, dh22}).Build()
			return
		}
		//        REGULAR-CASE..
		dp1 = dd1 * dx1Return
		dq2 = dp2 * dy1
		dq1 = dp1 * dx1Return
		//
		if math.Abs(dq1) > math.Abs(dq2) {
			dh21 = -dy1 / dx1Return
			dh12 = dp2 / dp1

			du = 1 - dh12*dh21

			if du > 0 {
				dflag = 0
				dd1Return /= du
				dd2Return /= du
				dx1Return *= du
			}
		} else {
			if dq2 < 0 {
				//              GO ZERO-H-D-AND-DX1..
				dflag = -1
				dh11 = 0
				dh12 = 0
				dh21 = 0
				dh22 = 0

				dd1Return = 0
				dd2Return = 0
				dx1Return = 0
			} else {
				dflag = 1
				dh11 = dp1 / dp2
				dh22 = dx1Return / dy1
				du = 1 + dh11*dh22
				dd1Return, dd2Return = dd2Return/du, dd1Return/du
				dx1Return = dy1 * du
			}
		}
		//     PROCEDURE..SCALE-CHECK
		if dd1Return != 0 {
			for (dd1Return <= rgamsq) || (dd1Return >= gamsq) {
				if dflag == 0 {
					dh11 = 1
					dh22 = 1
					dflag = -1
				} else {
					dh21 = -1
					dh12 = 1
					dflag = -1
				}
				if dd1Return <= rgamsq {
					dd1Return *= math.Pow(gam, 2)
					dx1Return /= gam
					dh11 /= gam
					dh12 /= gam
				} else {
					dd1Return /= math.Pow(gam, 2)
					dx1Return *= gam
					dh11 *= gam
					dh12 *= gam
				}
			}
		}
		if dd2Return != 0 {
			for (math.Abs(dd2Return) <= rgamsq) || (math.Abs(dd2Return) >= gamsq) {
				if dflag == 0 {
					dh11 = 1
					dh22 = 1
					dflag = -1
				} else {
					dh21 = -1
					dh12 = 1
					dflag = -1
				}
				if math.Abs(dd2Return) <= rgamsq {
					dd2Return *= math.Pow(gam, 2)
					dh21 /= gam
					dh22 /= gam
				} else {
					dd2Return /= math.Pow(gam, 2)
					dh21 *= gam
					dh22 *= gam
				}
			}
		}
	}
	if dflag < 0 {
		dparamReturn = dmb.Flag(dflag).H([4]float64{dh11, dh21, dh12, dh22}).Build()
	} else if dflag == 0 {
		dparamReturn = dmb.Flag(dflag).H([4]float64{0, dh21, dh12, 0}).Build()
	} else {
		dparamReturn = dmb.Flag(dflag).H([4]float64{dh11, 0, 0, dh22}).Build()
	}
	return
}

// Dscal scales a vector by a constant
func Dscal(n int, da float64, dx *mat.Vector, incx int) {
	if n <= 0 || incx <= 0 {
		return
	}

	blocksize := 512

	if n < minParBlocks*blocksize {
		dscal(n, da, dx, incx)
	} else {
		nblocks := blocks(n, blocksize)
		var wg sync.WaitGroup
		defer wg.Wait()

		for i := 0; i < nblocks; i++ {
			size := blocksize
			if (i+1)*blocksize > n {
				size = n - i*blocksize
			}
			wg.Add(1)
			go func(i, size int) {
				defer wg.Done()
				dscal(size, da, dx.Off(i*blocksize*incx), incx)
			}(i, size)
		}
	}
}
func dscal(n int, da float64, dx *mat.Vector, incx int) {

	for i := 0; i < n*incx; i += incx {
		dx.Set(i, da*dx.Get(i))
	}
}

// Dswap interchanges two vectors
func Dswap(n int, dx *mat.Vector, incx int, dy *mat.Vector, incy int) {
	if n <= 0 {
		return
	}

	blocksize := 512

	if n < minParBlocks*blocksize {
		dswap(n, dx, incx, dy, incy)
	} else {
		nblocks := blocks(n, blocksize)
		var wg sync.WaitGroup
		defer wg.Wait()

		for i := 0; i < nblocks; i++ {
			size := blocksize
			if (i+1)*blocksize > n {
				size = n - i*blocksize
			}
			wg.Add(1)
			go func(i, size int) {
				defer wg.Done()
				dswap(size, dx.Off(i*blocksize*incx), incx, dy.Off(i*blocksize*incy), incy)
			}(i, size)
		}
	}
}
func dswap(n int, dx *mat.Vector, incx int, dy *mat.Vector, incy int) {
	var dtemp float64
	var i, ix, iy int

	if incx < 0 {
		ix = (-n + 1) * incx
	}
	if incy < 0 {
		iy = (-n + 1) * incy
	}
	for i = 0; i < n; i, ix, iy = i+1, ix+incx, iy+incy {
		dtemp = dx.Get(ix)
		dx.Set(ix, dy.Get(iy))
		dy.Set(iy, dtemp)
	}
}

// Idamax finds the index of the first element having maximum absolute value
func Idamax(n int, dx *mat.Vector, incx int) (idamaxReturn int) {
	var dmax float64
	var i, ix int

	if n < 1 || incx <= 0 {
		return 0
	}
	if n == 1 {
		return 1
	}

	idamaxReturn = 1
	dmax = math.Abs(dx.Get(0))
	for i, ix = 1, incx; i < n; i, ix = i+1, ix+incx {
		if math.Abs(dx.Get(ix)) > dmax {
			idamaxReturn = i + 1
			dmax = math.Abs(dx.Get(ix))
		}
	}

	return
}
