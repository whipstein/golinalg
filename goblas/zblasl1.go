package goblas

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// dcabs1 computes |Re(.)| + |Im(.)| of a double complex number
func dcabs1(z complex128) (dcabs1Return float64) {

	dcabs1Return = math.Abs(real(z)) + math.Abs(imag(z))
	return
}

// Dzasum takes the sum of the (|Re(.)| + |Im(.)|)'s of a complex vector and
//    returns a single precision result.
func Dzasum(n int, zx *mat.CVector, incx int) (dzasumReturn float64) {
	// 	if n <= 0 || zx.Inc <= 0 {
	// 		return 0
	// 	}

	// 	blocksize := 512

	// 	if n < minParBlocks*blocksize {
	// 		return dzasum(n, zx)
	// 	}

	// 	nblocks := blocks(n, blocksize)
	// 	xout := make([]float64, nblocks)
	// 	var wg sync.WaitGroup

	// 	for i := 0; i < nblocks; i++ {
	// 		size := blocksize
	// 		if (i+1)*blocksize > n {
	// 			size = n - i*blocksize
	// 		}
	// 		wg.Add(1)
	// 		go func(i, size int) {
	// 			defer wg.Done()
	// 			xout[i] = dzasum(size, zx.Off(i*blocksize*zx.Inc))
	// 		}(i, size)
	// 	}
	// 	wg.Wait()

	// 	for _, val := range xout {
	// 		dzasumReturn += val
	// 	}

	// 	return
	// }
	// func dzasum(n int, zx *mat.CVector) (dzasumReturn float64) {
	// 	for _, ix := range zx.Iter(n) {
	// 		dzasumReturn += dcabs1(zx.Get(ix))
	// 	}

	return zx.Asum(n, incx)
}

// Dznrm2 returns the euclidean norm of a vector via the function
// name, so that
//
//    DZNRM2 := sqrt( x**H*x )
func Dznrm2(n int, x *mat.CVector, incx int) (dznrm2Return float64) {
	// 	if n < 1 || incx < 1 {
	// 		return 0
	// 	}

	// 	blocksize := 256

	// 	if n < minParBlocks*blocksize {
	// 		return dznrm2(n, x, incx)
	// 	}

	// 	nblocks := blocks(n, blocksize)
	// 	xout := make([]float64, nblocks)
	// 	var wg sync.WaitGroup

	// 	for i := 0; i < nblocks; i++ {
	// 		size := blocksize
	// 		if (i+1)*blocksize > n {
	// 			size = n - i*blocksize
	// 		}
	// 		wg.Add(1)
	// 		go func(i, size int) {
	// 			defer wg.Done()
	// 			xout[i] = dznrm2(size, x.Off(i*blocksize*incx), incx)
	// 		}(i, size)
	// 	}
	// 	wg.Wait()

	// 	for _, val := range xout {
	// 		dznrm2Return += val
	// 	}

	// 	return
	// }
	// func dznrm2(n int, x *mat.CVector, incx int) float64 {
	// 	var scale, ssq, temp float64
	// 	var ix int

	// 	xiter := x.Iter(n, incx)
	// 	ssq = 1
	// 	//        The following loop is equivalent to this call to the LAPACK
	// 	//        auxiliary routine:
	// 	//        CALL ZLASSQ( N, X, INCX, SCALE, SSQ )
	// 	for _, ix = range xiter {
	// 		if real(x.Get(ix)) != 0 {
	// 			temp = cmplx.Abs(x.GetReCmplx(ix))
	// 			if scale < temp {
	// 				ssq = 1 + ssq*math.Pow(scale/temp, 2)
	// 				scale = temp
	// 			} else {
	// 				ssq += math.Pow(temp/scale, 2)
	// 			}
	// 		}
	// 		if imag(x.Get(ix)) != 0 {
	// 			temp = cmplx.Abs(x.GetImCmplx(ix))
	// 			if scale < temp {
	// 				ssq = 1 + ssq*math.Pow(scale/temp, 2)
	// 				scale = temp
	// 			} else {
	// 				ssq += math.Pow(temp/scale, 2)
	// 			}
	// 		}
	// 	}

	// 	return scale * math.Sqrt(ssq)
	return x.Nrm2(n, incx)
}

// Izamax finds the index of the first element having maximum |Re(.)| + |Im(.)|
func Izamax(n int, zx *mat.CVector, incx int) (izamaxReturn int) {
	// var dmax float64

	// if n < 1 || incx <= 0 {
	// 	return 0
	// } else if n == 1 {
	// 	return 1
	// }

	// //        code for increment not equal to 1
	// izamaxReturn = 1
	// dmax = dcabs1(zx.Get(0))
	// for i, ix := range zx.Iter(n, incx) {
	// 	if dcabs1(zx.Get(ix)) > dmax {
	// 		izamaxReturn = i + 1
	// 		dmax = dcabs1(zx.Get(ix))
	// 	}
	// }
	return zx.Iamax(n, incx)
}

// Zdscal scales a vector by a constant.
func Zdscal(n int, da float64, zx *mat.CVector, incx int) {
	// 	if n <= 0 || incx < 0 {
	// 		return
	// 	}

	// 	blocksize := 512

	// 	if n < minParBlocks*blocksize {
	// 		zdscal(n, da, zx, incx)
	// 	} else {
	// 		nblocks := blocks(n, blocksize)
	// 		var wg sync.WaitGroup
	// 		defer wg.Wait()

	// 		for i := 0; i < nblocks; i++ {
	// 			size := blocksize
	// 			if (i+1)*blocksize > n {
	// 				size = n - i*blocksize
	// 			}
	// 			wg.Add(1)
	// 			go func(i, size int) {
	// 				defer wg.Done()
	// 				zdscal(size, da, zx.Off(i*blocksize*incx), incx)
	// 			}(i, size)
	// 		}
	// 	}
	// }
	// func zdscal(n int, da float64, zx *mat.CVector, incx int) {
	// 	for _, ix := range zx.Iter(n, incx) {
	// 		zx.Set(ix, complex(da, 0)*zx.Get(ix))
	// 	}
	zx.Dscal(n, da, incx)
}

// Zscal scales a vector by a constant.
func Zscal(n int, za complex128, zx *mat.CVector, incx int) {
	// 	if n <= 0 || incx < 0 {
	// 		return
	// 	}

	// 	blocksize := 128

	// 	if n < minParBlocks*blocksize {
	// 		zscal(n, za, zx, incx)
	// 	} else {
	// 		nblocks := blocks(n, blocksize)
	// 		var wg sync.WaitGroup
	// 		defer wg.Wait()

	// 		for i := 0; i < nblocks; i++ {
	// 			size := blocksize
	// 			if (i+1)*blocksize > n {
	// 				size = n - i*blocksize
	// 			}
	// 			wg.Add(1)
	// 			go func(i, size int) {
	// 				defer wg.Done()
	// 				zscal(size, za, zx.Off(i*blocksize*incx), incx)
	// 			}(i, size)
	// 		}
	// 	}
	// }
	// func zscal(n int, za complex128, zx *mat.CVector, incx int) {
	// 	for _, ix := range zx.Iter(n, incx) {
	// 		zx.Set(ix, za*zx.Get(ix))
	// 	}
	zx.Scal(n, za, incx)
}

// Zaxpy constant times a vector plus a vector.
func Zaxpy(n int, za complex128, zx *mat.CVector, incx int, zy *mat.CVector, incy int) {
	// 	if n <= 0 || dcabs1(za) == 0.0 {
	// 		return
	// 	}

	// 	blocksize := 256

	// 	if n < minParBlocks*blocksize {
	// 		zaxpy(n, za, zx, incx, zy, incy)
	// 	} else {
	// 		nblocks := blocks(n, blocksize)
	// 		var wg sync.WaitGroup
	// 		defer wg.Wait()

	// 		for i := 0; i < nblocks; i++ {
	// 			size := blocksize
	// 			if (i+1)*blocksize > n {
	// 				size = n - i*blocksize
	// 			}
	// 			wg.Add(1)
	// 			go func(i, size int) {
	// 				defer wg.Done()
	// 				zaxpy(size, za, zx.Off(i*blocksize*incx), incx, zy.Off(i*blocksize*incy), incy)
	// 			}(i, size)
	// 		}
	// 	}
	// }
	// func zaxpy(n int, za complex128, zx *mat.CVector, incx int, zy *mat.CVector, incy int) {
	// 	ix := zx.Iter(n, incx)
	// 	iy := zy.Iter(n, incy)

	// 	for i := 0; i < n; i++ {
	// 		zy.Set(iy[i], zy.Get(iy[i])+za*zx.Get(ix[i]))
	// 	}
	zy.Axpy(n, za, zx, incx, incy)
}

// Zcopy copies a vector, x, to a vector, y.
func Zcopy(n int, zx *mat.CVector, incx int, zy *mat.CVector, incy int) {
	// if n <= 0 {
	// 	return
	// }

	// ix := zx.Iter(n, incx)
	// iy := zy.Iter(n, incy)

	// for i := 0; i < n; i++ {
	// 	zy.Set(iy[i], zx.Get(ix[i]))
	// }
	zy.Copy(n, zx, incx, incy)
}

// Zdotc forms the dot product of two complex vectors
//      ZDOTC = X^H * Y
func Zdotc(n int, zx *mat.CVector, incx int, zy *mat.CVector, incy int) (zdotcReturn complex128) {
	// 	if n <= 0 {
	// 		return
	// 	}

	// 	blocksize := 256

	// 	if n < minParBlocks*blocksize {
	// 		return zdotc(n, zx, incx, zy, incy)
	// 	}

	// 	nblocks := blocks(n, blocksize)
	// 	x := make([]complex128, nblocks)
	// 	var wg sync.WaitGroup

	// 	for i := 0; i < nblocks; i++ {
	// 		size := blocksize
	// 		if (i+1)*blocksize > n {
	// 			size = n - i*blocksize
	// 		}
	// 		wg.Add(1)
	// 		go func(i, size int) {
	// 			defer wg.Done()
	// 			x[i] = zdotc(size, zx.Off(i*blocksize*incx), incx, zy.Off(i*blocksize*incy), incy)
	// 		}(i, size)
	// 	}
	// 	wg.Wait()

	// 	for _, val := range x {
	// 		zdotcReturn += val
	// 	}

	// 	return
	// }
	// func zdotc(n int, zx *mat.CVector, incx int, zy *mat.CVector, incy int) (zdotcReturn complex128) {
	// 	ix := zx.Iter(n, incx)
	// 	iy := zy.Iter(n, incy)

	// 	for i := 0; i < n; i++ {
	// 		zdotcReturn += zx.GetConj(ix[i]) * zy.Get(iy[i])
	// 	}

	return zy.Dotc(n, zx, incx, incy)
}

// Zdotu forms the dot product of two complex vectors
//      ZDOTU = X^T * Y
func Zdotu(n int, zx *mat.CVector, incx int, zy *mat.CVector, incy int) (zdotuReturn complex128) {
	// 	if n <= 0 {
	// 		return
	// 	}

	// 	blocksize := 256

	// 	if n < minParBlocks*blocksize {
	// 		return zdotu(n, zx, incx, zy, incy)
	// 	}

	// 	nblocks := blocks(n, blocksize)
	// 	x := make([]complex128, nblocks)
	// 	var wg sync.WaitGroup

	// 	for i := 0; i < nblocks; i++ {
	// 		size := blocksize
	// 		if (i+1)*blocksize > n {
	// 			size = n - i*blocksize
	// 		}
	// 		wg.Add(1)
	// 		go func(i, size int) {
	// 			defer wg.Done()
	// 			x[i] = zdotu(size, zx.Off(i*blocksize*incx), incx, zy.Off(i*blocksize*incy), incy)
	// 		}(i, size)
	// 	}
	// 	wg.Wait()

	// 	for _, val := range x {
	// 		zdotuReturn += val
	// 	}

	// 	return
	// }
	// func zdotu(n int, zx *mat.CVector, incx int, zy *mat.CVector, incy int) (zdotuReturn complex128) {
	// 	ix := zx.Iter(n, incx)
	// 	iy := zy.Iter(n, incy)

	// 	for i := 0; i < n; i++ {
	// 		zdotuReturn += zx.Get(ix[i]) * zy.Get(iy[i])
	// 	}

	return zy.Dotu(n, zx, incx, incy)
}

// Zswap interchanges two vectors.
func Zswap(n int, zx *mat.CVector, incx int, zy *mat.CVector, incy int) {
	// 	if n <= 0 {
	// 		return
	// 	}

	// 	blocksize := 512

	// 	if n < minParBlocks*blocksize {
	// 		zswap(n, zx, incx, zy, incy)
	// 	} else {
	// 		nblocks := blocks(n, blocksize)
	// 		var wg sync.WaitGroup
	// 		defer wg.Wait()

	// 		for i := 0; i < nblocks; i++ {
	// 			size := blocksize
	// 			if (i+1)*blocksize > n {
	// 				size = n - i*blocksize
	// 			}
	// 			wg.Add(1)
	// 			go func(i, size int) {
	// 				defer wg.Done()
	// 				zswap(size, zx.Off(i*blocksize*incx), incx, zy.Off(i*blocksize*incy), incy)
	// 			}(i, size)
	// 		}
	// 	}
	// }
	// func zswap(n int, zx *mat.CVector, incx int, zy *mat.CVector, incy int) {
	// 	var ztemp complex128

	// 	ix := zx.Iter(n, incx)
	// 	iy := zy.Iter(n, incy)

	// 	for i := 0; i < n; i++ {
	// 		ztemp = zx.Get(ix[i])
	// 		zx.Set(ix[i], zy.Get(iy[i]))
	// 		zy.Set(iy[i], ztemp)
	// 	}
	zy.Swap(n, zx, incx, incy)
}

// Zdrot Applies a plane rotation, where the cos and sin (c and s) are real
// and the vectors cx and cy are complex.
// jack dongarra, linpack, 3/11/78.
func Zdrot(n int, cx *mat.CVector, incx int, cy *mat.CVector, incy int, c, s float64) {
	// 	if n <= 0 {
	// 		return
	// 	}

	// 	blocksize := 512

	// 	if n < minParBlocks*blocksize {
	// 		zdrot(n, cx, incx, cy, incy, c, s)
	// 	} else {
	// 		nblocks := blocks(n, blocksize)
	// 		var wg sync.WaitGroup
	// 		defer wg.Wait()

	// 		for i := 0; i < nblocks; i++ {
	// 			size := blocksize
	// 			if (i+1)*blocksize > n {
	// 				size = n - i*blocksize
	// 			}
	// 			wg.Add(1)
	// 			go func(i, size int) {
	// 				defer wg.Done()
	// 				zdrot(size, cx.Off(i*blocksize*incx), incx, cy.Off(i*blocksize*incy), incy, c, s)
	// 			}(i, size)
	// 		}
	// 	}
	// }
	// func zdrot(n int, cx *mat.CVector, incx int, cy *mat.CVector, incy int, c, s float64) {
	// 	var ctemp complex128

	// 	ix := cx.Iter(n, incx)
	// 	iy := cy.Iter(n, incy)

	// 	for i := 0; i < n; i++ {
	// 		ctemp = complex(c, 0)*cx.Get(ix[i]) + complex(s, 0)*cy.Get(iy[i])
	// 		cy.Set(iy[i], complex(c, 0)*cy.Get(iy[i])-complex(s, 0)*cx.Get(ix[i]))
	// 		cx.Set(ix[i], ctemp)
	// 	}
	cy.Drot(n, cx, incx, incy, c, s)
}

// Zrotg determines a double complex Givens rotation.
func Zrotg(ca, cb complex128, c float64, s complex128) (cReturn float64, sReturn, caReturn complex128) {
	if cmplx.Abs(ca) == 0.0 {
		return 0.0, 1.0 + 0.0i, cb
	}

	scale := cmplx.Abs(ca) + cmplx.Abs(cb)
	norm := scale * math.Sqrt(math.Pow(cmplx.Abs((ca)/complex(scale, 0)), 2)+math.Pow(cmplx.Abs((cb)/complex(scale, 0)), 2))
	alpha := ca / complex(cmplx.Abs(ca), 0)
	cReturn = cmplx.Abs(ca) / norm
	sReturn = alpha * cmplx.Conj(cb) / complex(norm, 0)
	caReturn = alpha * complex(norm, 0)

	return
}
