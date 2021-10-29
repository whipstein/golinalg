package matgen

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlatme generates random non-symmetric square matrices with
//    specified eigenvalues for testing LAPACK programs.
//
//    Dlatme operates by applying the following sequence of
//    operations:
//
//    1. Set the diagonal to D, where D may be input or
//         computed according to MODE, COND, DMAX, and RSIGN
//         as described below.
//
//    2. If complex conjugate pairs are desired (MODE=0 and EI(1)='R',
//         or MODE=5), certain pairs of adjacent elements of D are
//         interpreted as the real and complex parts of a complex
//         conjugate pair; A thus becomes block diagonal, with 1x1
//         and 2x2 blocks.
//
//    3. If UPPER='T', the upper triangle of A is set to random values
//         out of distribution DIST.
//
//    4. If SIM='T', A is multiplied on the left by a random matrix
//         X, whose singular values are specified by DS, MODES, and
//         CONDS, and on the right by X inverse.
//
//    5. If KL < N-1, the lower bandwidth is reduced to KL using
//         Householder transformations.  If KU < N-1, the upper
//         bandwidth is reduced to KU.
//
//    6. If ANORM is not negative, the matrix is scaled to have
//         maximum-element-norm ANORM.
//
//    (Note: since the matrix cannot be reduced beyond Hessenberg form,
//     no packing options are available.)
func Dlatme(n int, dist byte, iseed *[]int, d *mat.Vector, mode int, cond, dmax float64, ei []byte, rsign, upper, sim byte, ds *mat.Vector, modes int, conds float64, kl, ku int, anorm float64, a *mat.Matrix, work *mat.Vector) (info int, err error) {
	var badei, bads, useei bool
	var alpha, half, one, tau, temp, xnorms, zero float64
	var i, ic, icols, idist, ir, irows, irsign, isim, iupper, j, jc, jcr, jr int

	tempa := vf(1)

	zero = 0.0
	one = 1.0
	half = 1.0 / 2.0

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
	} else {
		idist = -1
	}

	//     Check EI
	useei = true
	badei = false
	if ei[0] == ' ' || mode != 0 {
		useei = false
	} else {
		if ei[0] == 'R' {
			for j = 2; j <= n; j++ {
				if ei[j-1] == 'I' {
					if ei[j-1-1] == 'I' {
						badei = true
					}
				} else {
					if ei[j-1] != 'R' {
						badei = true
					}
				}
			}
		} else {
			badei = true
		}
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
		info = -1
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if idist == -1 {
		info = -2
		err = fmt.Errorf("idist == -1: dist='%c'", dist)
	} else if abs(mode) > 6 {
		info = -5
		err = fmt.Errorf("abs(mode) > 6: mode=%v", mode)
	} else if (mode != 0 && abs(mode) != 6) && cond < one {
		info = -6
		err = fmt.Errorf("(mode != 0 && abs(mode) != 6) && cond < one: mode=%v, cond=%v", mode, cond)
	} else if badei {
		info = -8
		err = fmt.Errorf("badei: badei=%v", badei)
	} else if irsign == -1 {
		info = -9
		err = fmt.Errorf("irsign == -1: rsign='%c'", rsign)
	} else if iupper == -1 {
		info = -10
		err = fmt.Errorf("iupper == -1: upper='%c'", upper)
	} else if isim == -1 {
		info = -11
		err = fmt.Errorf("isim == -1: sim='%c'", sim)
	} else if bads {
		info = -12
		err = fmt.Errorf("bads: bads=%v", bads)
	} else if isim == 1 && abs(modes) > 5 {
		info = -13
		err = fmt.Errorf("isim == 1 && abs(modes) > 5: sim='%c', modes=%v", sim, modes)
	} else if isim == 1 && modes != 0 && conds < one {
		info = -14
		err = fmt.Errorf("isim == 1 && modes != 0 && conds < one: sim='%c', modes=%v, conds=%v", sim, modes, conds)
	} else if kl < 1 {
		info = -15
		err = fmt.Errorf("kl < 1: kl=%v", kl)
	} else if ku < 1 || (ku < n-1 && kl < n-1) {
		info = -16
		err = fmt.Errorf("ku < 1 || (ku < n-1 && kl < n-1): kl=%v, ku=%v, n=%v", kl, ku, n)
	} else if a.Rows < max(1, n) {
		info = -19
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}

	if err != nil {
		gltest.Xerbla2("Dlatme", err)
		return
	}

	//     Initialize random number generator
	for i = 1; i <= 4; i++ {
		(*iseed)[i-1] = abs((*iseed)[i-1]) % 4096
	}

	if ((*iseed)[3] % 2) != 1 {
		(*iseed)[3] = (*iseed)[3] + 1
	}

	//     2)      Set up diagonal of A
	//
	//             Compute D according to COND and MODE
	if err = Dlatm1(mode, cond, irsign, idist, iseed, d, n); err != nil {
		panic(err)
	}
	if mode != 0 && abs(mode) != 6 {
		//        Scale by DMAX
		temp = math.Abs(d.Get(0))
		for i = 2; i <= n; i++ {
			temp = math.Max(temp, math.Abs(d.Get(i-1)))
		}

		if temp > zero {
			alpha = dmax / temp
		} else if dmax != zero {
			info = 2
			return
		} else {
			alpha = zero
		}

		goblas.Dscal(n, alpha, d.Off(0, 1))

	}

	golapack.Dlaset(Full, n, n, zero, zero, a)
	goblas.Dcopy(n, d.Off(0, 1), a.VectorIdx(0, a.Rows+1))

	//     Set up complex conjugate pairs
	if mode == 0 {
		if useei {
			for j = 2; j <= n; j++ {
				if ei[j-1] == 'I' {
					a.Set(j-1-1, j-1, a.Get(j-1, j-1))
					a.Set(j-1, j-1-1, -a.Get(j-1, j-1))
					a.Set(j-1, j-1, a.Get(j-1-1, j-1-1))
				}
			}
		}

	} else if abs(mode) == 5 {

		for j = 2; j <= n; j += 2 {
			if Dlaran(iseed) > half {
				a.Set(j-1-1, j-1, a.Get(j-1, j-1))
				a.Set(j-1, j-1-1, -a.Get(j-1, j-1))
				a.Set(j-1, j-1, a.Get(j-1-1, j-1-1))
			}
		}
	}

	//     3)      If UPPER='T', set upper triangle of A to random numbers.
	//             (but don't modify the corners of 2x2 blocks.)
	if iupper != 0 {
		for jc = 2; jc <= n; jc++ {
			if a.Get(jc-1-1, jc-1) != zero {
				jr = jc - 2
			} else {
				jr = jc - 1
			}
			golapack.Dlarnv(idist, iseed, jr, a.Vector(0, jc-1))
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
			info = 3
			return
		}

		//        Multiply by V and V'
		if err = Dlarge(n, a, iseed, work); err != nil {
			info = 4
			return
		}

		//        Multiply by S and (1/S)
		for j = 1; j <= n; j++ {
			goblas.Dscal(n, ds.Get(j-1), a.Vector(j-1, 0))
			if ds.Get(j-1) != zero {
				goblas.Dscal(n, one/ds.Get(j-1), a.Vector(0, j-1, 1))
			} else {
				info = 5
				return
			}
		}

		//        Multiply by U and U'
		if err = Dlarge(n, a, iseed, work); err != nil {
			info = 4
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

			goblas.Dcopy(irows, a.Vector(jcr-1, ic-1, 1), work.Off(0, 1))
			xnorms = work.Get(0)
			xnorms, tau = golapack.Dlarfg(irows, xnorms, work.Off(1, 1))
			work.Set(0, one)

			err = goblas.Dgemv(Trans, irows, icols, one, a.Off(jcr-1, ic), work.Off(0, 1), zero, work.Off(irows, 1))
			err = goblas.Dger(irows, icols, -tau, work.Off(0, 1), work.Off(irows, 1), a.Off(jcr-1, ic))

			err = goblas.Dgemv(NoTrans, n, irows, one, a.Off(0, jcr-1), work.Off(0, 1), zero, work.Off(irows, 1))
			err = goblas.Dger(n, irows, -tau, work.Off(irows, 1), work.Off(0, 1), a.Off(0, jcr-1))

			a.Set(jcr-1, ic-1, xnorms)
			golapack.Dlaset(Full, irows-1, 1, zero, zero, a.Off(jcr, ic-1))
		}
	} else if ku < n-1 {
		//        Reduce upper bandwidth -- kill a row at a time.
		for jcr = ku + 1; jcr <= n-1; jcr++ {
			ir = jcr - ku
			irows = n + ku - jcr
			icols = n + 1 - jcr

			goblas.Dcopy(icols, a.Vector(ir-1, jcr-1), work.Off(0, 1))
			xnorms = work.Get(0)
			xnorms, tau = golapack.Dlarfg(icols, xnorms, work.Off(1, 1))
			work.Set(0, one)

			err = goblas.Dgemv(NoTrans, irows, icols, one, a.Off(ir, jcr-1), work.Off(0, 1), zero, work.Off(icols, 1))
			err = goblas.Dger(irows, icols, -tau, work.Off(icols, 1), work.Off(0, 1), a.Off(ir, jcr-1))

			err = goblas.Dgemv(ConjTrans, icols, n, one, a.Off(jcr-1, 0), work.Off(0, 1), zero, work.Off(icols, 1))
			err = goblas.Dger(icols, n, -tau, work.Off(0, 1), work.Off(icols, 1), a.Off(jcr-1, 0))

			a.Set(ir-1, jcr-1, xnorms)
			golapack.Dlaset(Full, 1, icols-1, zero, zero, a.Off(ir-1, jcr))
		}
	}

	//     Scale the matrix to have norm ANORM
	if anorm >= zero {
		temp = golapack.Dlange('M', n, n, a, tempa)
		if temp > zero {
			alpha = anorm / temp
			for j = 1; j <= n; j++ {
				goblas.Dscal(n, alpha, a.Vector(0, j-1, 1))
			}
		}
	}

	return
}
