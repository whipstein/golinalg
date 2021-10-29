package matgen

import (
	"fmt"
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
//    Zlatme operates by applying the following sequence of
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
func Zlatme(n int, dist byte, iseed *[]int, d *mat.CVector, mode int, cond float64, dmax complex128, rsign, upper, sim byte, ds *mat.Vector, modes int, conds float64, kl, ku int, anorm float64, a *mat.CMatrix, work *mat.CVector) (err error) {
	var bads bool
	var alpha, cone, czero, tau, xnorms complex128
	var one, ralpha, temp, zero float64
	var i, ic, icols, idist, ir, irows, irsign, isim, iupper, j, jc, jcr int

	tempa := vf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     1)      Decode and Test the input parameters.
	//             Initialize flags & seed.
	//     Quick return if possible
	if n == 0 {
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
	if modes == 0 && isim == 1 {
		for j = 1; j <= n; j++ {
			if ds.Get(j-1) == zero {
				bads = true
			}
		}
	}

	//     Set INFO if an error
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if idist == -1 {
		err = fmt.Errorf("idist == -1: dist='%c'", dist)
	} else if abs(mode) > 6 {
		err = fmt.Errorf("abs(mode) > 6: mode=%v", mode)
	} else if (mode != 0 && abs(mode) != 6) && cond < one {
		err = fmt.Errorf("(mode != 0 && abs(mode) != 6) && cond < one: mode=%v, cond=%v", mode, cond)
	} else if irsign == -1 {
		err = fmt.Errorf("irsign == -1: rsign=%v", rsign)
	} else if iupper == -1 {
		err = fmt.Errorf("iupper == -1: upper='%c'", upper)
	} else if isim == -1 {
		err = fmt.Errorf("isim == -1: sim='%c'", sim)
	} else if bads {
		err = fmt.Errorf("bads: ds=%v", ds.Data)
	} else if isim == 1 && abs(modes) > 5 {
		err = fmt.Errorf("isim == 1 && abs(modes) > 5: sim='%c', modes=%v", sim, modes)
	} else if isim == 1 && modes != 0 && conds < one {
		err = fmt.Errorf("isim == 1 && modes != 0 && conds < one: sim='%c', modes=%v, conds=%v", sim, modes, conds)
	} else if kl < 1 {
		err = fmt.Errorf("kl < 1: kl=%v", kl)
	} else if ku < 1 || (ku < n-1 && kl < n-1) {
		err = fmt.Errorf("ku < 1 || (ku < n-1 && kl < n-1): kl=%v, ku=%v, n=%v", kl, ku, n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}

	if err != nil {
		gltest.Xerbla2("Zlatme", err)
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
	if err = Zlatm1(mode, cond, irsign, idist, iseed, d, n); err != nil {
		err = fmt.Errorf("Error return from Zlatm1 (computing D)")
		return
	}
	if mode != 0 && abs(mode) != 6 {
		//        Scale by DMAX
		temp = d.GetMag(0)
		for i = 2; i <= n; i++ {
			temp = math.Max(temp, d.GetMag(i-1))
		}

		if temp > zero {
			alpha = dmax / complex(temp, 0)
		} else {
			err = fmt.Errorf("Cannot scale to dmax (max. eigenvalue is 0)")
			return
		}

		goblas.Zscal(n, alpha, d.Off(0, 1))

	}

	golapack.Zlaset(Full, n, n, czero, czero, a)
	goblas.Zcopy(n, d.Off(0, 1), a.CVector(0, 0, a.Rows+1))

	//     3)      If UPPER='T', set upper triangle of A to random numbers.
	if iupper != 0 {
		for jc = 2; jc <= n; jc++ {
			golapack.Zlarnv(idist, iseed, jc-1, a.CVector(0, jc-1))
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
		if err = Dlatm1(modes, conds, 0, 0, iseed, ds, n); err != nil {
			err = fmt.Errorf("Error return from Dlatm1 (computing DS)")
			return
		}

		//        Multiply by V and V'
		if err = Zlarge(n, a, iseed, work); err != nil {
			err = fmt.Errorf("Error return from Zlarge")
			return
		}

		//        Multiply by S and (1/S)
		for j = 1; j <= n; j++ {
			goblas.Zdscal(n, ds.Get(j-1), a.CVector(j-1, 0))
			if ds.Get(j-1) != zero {
				goblas.Zdscal(n, one/ds.Get(j-1), a.CVector(0, j-1, 1))
			} else {
				err = fmt.Errorf("Zero singular value from Dlatm1")
				return
			}
		}

		//        Multiply by U and U'
		if err = Zlarge(n, a, iseed, work); err != nil {
			err = fmt.Errorf("Error return from Zlarge")
			return
		}
	}

	//     5)      Reduce the bandwidth.
	if kl < n-1 {
		//        Reduce bandwidth -- kill column
		for jcr = kl + 1; jcr <= n-1; jcr++ {
			ic = jcr - kl
			irows = n + 1 - jcr
			icols = n + kl - jcr

			goblas.Zcopy(irows, a.CVector(jcr-1, ic-1, 1), work.Off(0, 1))
			xnorms = work.Get(0)
			xnorms, tau = golapack.Zlarfg(irows, xnorms, work.Off(1, 1))
			tau = cmplx.Conj(tau)
			work.Set(0, cone)
			alpha = Zlarnd(5, *iseed)

			if err = goblas.Zgemv(ConjTrans, irows, icols, cone, a.Off(jcr-1, ic), work.Off(0, 1), czero, work.Off(irows, 1)); err != nil {
				panic(err)
			}
			if err = goblas.Zgerc(irows, icols, -tau, work.Off(0, 1), work.Off(irows, 1), a.Off(jcr-1, ic)); err != nil {
				panic(err)
			}

			if err = goblas.Zgemv(NoTrans, n, irows, cone, a.Off(0, jcr-1), work.Off(0, 1), czero, work.Off(irows, 1)); err != nil {
				panic(err)
			}
			if err = goblas.Zgerc(n, irows, -cmplx.Conj(tau), work.Off(irows, 1), work.Off(0, 1), a.Off(0, jcr-1)); err != nil {
				panic(err)
			}

			a.Set(jcr-1, ic-1, xnorms)
			golapack.Zlaset(Full, irows-1, 1, czero, czero, a.Off(jcr, ic-1))

			goblas.Zscal(icols+1, alpha, a.CVector(jcr-1, ic-1))
			goblas.Zscal(n, cmplx.Conj(alpha), a.CVector(0, jcr-1, 1))
		}
	} else if ku < n-1 {
		//        Reduce upper bandwidth -- kill a row at a time.
		for jcr = ku + 1; jcr <= n-1; jcr++ {
			ir = jcr - ku
			irows = n + ku - jcr
			icols = n + 1 - jcr

			goblas.Zcopy(icols, a.CVector(ir-1, jcr-1), work.Off(0, 1))
			xnorms = work.Get(0)
			xnorms, tau = golapack.Zlarfg(icols, xnorms, work.Off(1, 1))
			tau = cmplx.Conj(tau)
			work.Set(0, cone)
			golapack.Zlacgv(icols-1, work.Off(1, 1))
			alpha = Zlarnd(5, *iseed)

			if err = goblas.Zgemv(NoTrans, irows, icols, cone, a.Off(ir, jcr-1), work.Off(0, 1), czero, work.Off(icols, 1)); err != nil {
				panic(err)
			}
			if err = goblas.Zgerc(irows, icols, -tau, work.Off(icols, 1), work.Off(0, 1), a.Off(ir, jcr-1)); err != nil {
				panic(err)
			}

			if err = goblas.Zgemv(ConjTrans, icols, n, cone, a.Off(jcr-1, 0), work.Off(0, 1), czero, work.Off(icols, 1)); err != nil {
				panic(err)
			}
			if err = goblas.Zgerc(icols, n, -cmplx.Conj(tau), work.Off(0, 1), work.Off(icols, 1), a.Off(jcr-1, 0)); err != nil {
				panic(err)
			}

			a.Set(ir-1, jcr-1, xnorms)
			golapack.Zlaset(Full, 1, icols-1, czero, czero, a.Off(ir-1, jcr))

			goblas.Zscal(irows+1, alpha, a.CVector(ir-1, jcr-1, 1))
			goblas.Zscal(n, cmplx.Conj(alpha), a.CVector(jcr-1, 0))
		}
	}

	//     Scale the matrix to have norm ANORM
	if anorm >= zero {
		temp = golapack.Zlange('M', n, n, a, tempa)
		if temp > zero {
			ralpha = anorm / temp
			for j = 1; j <= n; j++ {
				goblas.Zdscal(n, ralpha, a.CVector(0, j-1, 1))
			}
		}
	}

	return
}
