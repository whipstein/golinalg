package matgen

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlatme generates random non-symmetric square matrices with
//    specified eigenvalues for testing LAPACK programs.
//
//    ZLATME operates by applying the following sequence of
//    operations:
//
//    1. Set the diagonal to D, where D may be input or
//         computed according to MODE, COND, DMAX, and RSIGN
//         as described below.
//
//    2. If UPPER='T', the upper triangle of A is set to random values
//         out of distribution DIST.
//
//    3. If SIM='T', A is multiplied on the left by a random matrix
//         X, whose singular values are specified by DS, MODES, and
//         CONDS, and on the right by X inverse.
//
//    4. If KL < N-1, the lower bandwidth is reduced to KL using
//         Householder transformations.  If KU < N-1, the upper
//         bandwidth is reduced to KU.
//
//    5. If ANORM is not negative, the matrix is scaled to have
//         maximum-element-norm ANORM.
//
//    (Note: since the matrix cannot be reduced beyond Hessenberg form,
//     no packing options are available.)
func Zlatme(n *int, dist byte, iseed *[]int, d *mat.CVector, mode *int, cond *float64, dmax *complex128, rsign, upper, sim byte, ds *mat.Vector, modes *int, conds *float64, kl, ku *int, anorm *float64, a *mat.CMatrix, lda *int, work *mat.CVector, info *int) {
	var bads bool
	var alpha, cone, czero, tau, xnorms complex128
	var one, ralpha, temp, zero float64
	var i, ic, icols, idist, iinfo, ir, irows, irsign, isim, iupper, j, jc, jcr int
	var err error
	_ = err

	tempa := vf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     1)      Decode and Test the input parameters.
	//             Initialize flags & seed.
	(*info) = 0

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Decode DIST
	if dist == 'U' {
		idist = 1
	} else if dist == 'S' {
		idist = 2
	} else if dist == 'N' {
		idist = 3
	} else if dist == 'D' {
		idist = 4
	} else {
		idist = -1
	}

	//     Decode RSIGN
	if rsign == 'T' {
		irsign = 1
	} else if rsign == 'F' {
		irsign = 0
	} else {
		irsign = -1
	}

	//     Decode UPPER
	if upper == 'T' {
		iupper = 1
	} else if upper == 'F' {
		iupper = 0
	} else {
		iupper = -1
	}

	//     Decode SIM
	if sim == 'T' {
		isim = 1
	} else if sim == 'F' {
		isim = 0
	} else {
		isim = -1
	}

	//     Check DS, if MODES=0 and ISIM=1
	bads = false
	if (*modes) == 0 && isim == 1 {
		for j = 1; j <= (*n); j++ {
			if ds.Get(j-1) == zero {
				bads = true
			}
		}
	}

	//     Set INFO if an error
	if (*n) < 0 {
		(*info) = -1
	} else if idist == -1 {
		(*info) = -2
	} else if abs(*mode) > 6 {
		(*info) = -5
	} else if ((*mode) != 0 && abs(*mode) != 6) && (*cond) < one {
		(*info) = -6
	} else if irsign == -1 {
		(*info) = -9
	} else if iupper == -1 {
		(*info) = -10
	} else if isim == -1 {
		(*info) = -11
	} else if bads {
		(*info) = -12
	} else if isim == 1 && abs(*modes) > 5 {
		(*info) = -13
	} else if isim == 1 && (*modes) != 0 && (*conds) < one {
		(*info) = -14
	} else if (*kl) < 1 {
		(*info) = -15
	} else if (*ku) < 1 || ((*ku) < (*n)-1 && (*kl) < (*n)-1) {
		(*info) = -16
	} else if (*lda) < max(1, *n) {
		(*info) = -19
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZLATME"), -(*info))
		return
	}

	//     Initialize random number generator
	for i = 1; i <= 4; i++ {
		(*iseed)[i-1] = abs((*iseed)[i-1] % 4096)
	}

	if ((*iseed)[3] % 2) != 1 {
		(*iseed)[3] = (*iseed)[3] + 1
	}

	//     2)      Set up diagonal of A
	//
	//             Compute D according to COND and MODE
	Zlatm1(mode, cond, &irsign, &idist, iseed, d, n, &iinfo)
	if iinfo != 0 {
		(*info) = 1
		return
	}
	if (*mode) != 0 && abs(*mode) != 6 {
		//        Scale by DMAX
		temp = d.GetMag(0)
		for i = 2; i <= (*n); i++ {
			temp = math.Max(temp, d.GetMag(i-1))
		}

		if temp > zero {
			alpha = (*dmax) / complex(temp, 0)
		} else {
			(*info) = 2
			return
		}

		goblas.Zscal(*n, alpha, d.Off(0, 1))

	}

	golapack.Zlaset('F', n, n, &czero, &czero, a, lda)
	goblas.Zcopy(*n, d.Off(0, 1), a.CVector(0, 0, (*lda)+1))

	//     3)      If UPPER='T', set upper triangle of A to random numbers.
	if iupper != 0 {
		for jc = 2; jc <= (*n); jc++ {
			golapack.Zlarnv(&idist, iseed, toPtr(jc-1), a.CVector(0, jc-1))
		}
	}

	//     4)      If SIM='T', apply similarity transformation.
	//
	//                                -1
	//             Transform is  X A X  , where X = U S V, thus
	//
	//             it is  U S V A V' (1/S) U'
	if isim != 0 {
		//        Compute S (singular values of the eigenvector matrix)
		//        according to CONDS and MODES
		Dlatm1(modes, conds, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), iseed, ds, n, &iinfo)
		if iinfo != 0 {
			(*info) = 3
			return
		}

		//        Multiply by V and V'
		Zlarge(n, a, lda, iseed, work, &iinfo)
		if iinfo != 0 {
			(*info) = 4
			return
		}

		//        Multiply by S and (1/S)
		for j = 1; j <= (*n); j++ {
			goblas.Zdscal(*n, ds.Get(j-1), a.CVector(j-1, 0, *lda))
			if ds.Get(j-1) != zero {
				goblas.Zdscal(*n, one/ds.Get(j-1), a.CVector(0, j-1, 1))
			} else {
				(*info) = 5
				return
			}
		}

		//        Multiply by U and U'
		Zlarge(n, a, lda, iseed, work, &iinfo)
		if iinfo != 0 {
			(*info) = 4
			return
		}
	}

	//     5)      Reduce the bandwidth.
	if (*kl) < (*n)-1 {
		//        Reduce bandwidth -- kill column
		for jcr = (*kl) + 1; jcr <= (*n)-1; jcr++ {
			ic = jcr - (*kl)
			irows = (*n) + 1 - jcr
			icols = (*n) + (*kl) - jcr

			goblas.Zcopy(irows, a.CVector(jcr-1, ic-1, 1), work.Off(0, 1))
			xnorms = work.Get(0)
			golapack.Zlarfg(&irows, &xnorms, work.Off(1), func() *int { y := 1; return &y }(), &tau)
			tau = cmplx.Conj(tau)
			work.Set(0, cone)
			alpha = Zlarnd(func() *int { y := 5; return &y }(), iseed)

			err = goblas.Zgemv(ConjTrans, irows, icols, cone, a.Off(jcr-1, ic), work.Off(0, 1), czero, work.Off(irows, 1))
			err = goblas.Zgerc(irows, icols, -tau, work.Off(0, 1), work.Off(irows, 1), a.Off(jcr-1, ic))

			err = goblas.Zgemv(NoTrans, *n, irows, cone, a.Off(0, jcr-1), work.Off(0, 1), czero, work.Off(irows, 1))
			err = goblas.Zgerc(*n, irows, -cmplx.Conj(tau), work.Off(irows, 1), work.Off(0, 1), a.Off(0, jcr-1))

			a.Set(jcr-1, ic-1, xnorms)
			golapack.Zlaset('F', toPtr(irows-1), func() *int { y := 1; return &y }(), &czero, &czero, a.Off(jcr, ic-1), lda)

			goblas.Zscal(icols+1, alpha, a.CVector(jcr-1, ic-1, *lda))
			goblas.Zscal(*n, cmplx.Conj(alpha), a.CVector(0, jcr-1, 1))
		}
	} else if (*ku) < (*n)-1 {
		//        Reduce upper bandwidth -- kill a row at a time.
		for jcr = (*ku) + 1; jcr <= (*n)-1; jcr++ {
			ir = jcr - (*ku)
			irows = (*n) + (*ku) - jcr
			icols = (*n) + 1 - jcr

			goblas.Zcopy(icols, a.CVector(ir-1, jcr-1, *lda), work.Off(0, 1))
			xnorms = work.Get(0)
			golapack.Zlarfg(&icols, &xnorms, work.Off(1), func() *int { y := 1; return &y }(), &tau)
			tau = cmplx.Conj(tau)
			work.Set(0, cone)
			golapack.Zlacgv(toPtr(icols-1), work.Off(1), func() *int { y := 1; return &y }())
			alpha = Zlarnd(func() *int { y := 5; return &y }(), iseed)

			err = goblas.Zgemv(NoTrans, irows, icols, cone, a.Off(ir, jcr-1), work.Off(0, 1), czero, work.Off(icols, 1))
			err = goblas.Zgerc(irows, icols, -tau, work.Off(icols, 1), work.Off(0, 1), a.Off(ir, jcr-1))

			err = goblas.Zgemv(ConjTrans, icols, *n, cone, a.Off(jcr-1, 0), work.Off(0, 1), czero, work.Off(icols, 1))
			err = goblas.Zgerc(icols, *n, -cmplx.Conj(tau), work.Off(0, 1), work.Off(icols, 1), a.Off(jcr-1, 0))

			a.Set(ir-1, jcr-1, xnorms)
			golapack.Zlaset('F', func() *int { y := 1; return &y }(), toPtr(icols-1), &czero, &czero, a.Off(ir-1, jcr), lda)

			goblas.Zscal(irows+1, alpha, a.CVector(ir-1, jcr-1, 1))
			goblas.Zscal(*n, cmplx.Conj(alpha), a.CVector(jcr-1, 0, *lda))
		}
	}

	//     Scale the matrix to have norm ANORM
	if (*anorm) >= zero {
		temp = golapack.Zlange('M', n, n, a, lda, tempa)
		if temp > zero {
			ralpha = (*anorm) / temp
			for j = 1; j <= (*n); j++ {
				goblas.Zdscal(*n, ralpha, a.CVector(0, j-1, 1))
			}
		}
	}
}
