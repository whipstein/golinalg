package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunbdb simultaneously bidiagonalizes the blocks of an M-by-M
// partitioned unitary matrix X:
//
//                                 [ B11 | B12 0  0 ]
//     [ X11 | X12 ]   [ P1 |    ] [  0  |  0 -I  0 ] [ Q1 |    ]**H
// X = [-----------] = [---------] [----------------] [---------]   .
//     [ X21 | X22 ]   [    | P2 ] [ B21 | B22 0  0 ] [    | Q2 ]
//                                 [  0  |  0  0  I ]
//
// X11 is P-by-Q. Q must be no larger than P, M-P, or M-Q. (If this is
// not the case, then X must be transposed and/or permuted. This can be
// done in constant time using the TRANS and SIGNS options. See ZUNCSD
// for details.)
//
// The unitary matrices P1, P2, Q1, and Q2 are P-by-P, (M-P)-by-
// (M-P), Q-by-Q, and (M-Q)-by-(M-Q), respectively. They are
// represented implicitly by Householder vectors.
//
// B11, B12, B21, and B22 are Q-by-Q bidiagonal matrices represented
// implicitly by angles THETA, PHI.
func Zunbdb(trans mat.MatTrans, signs byte, m, p, q int, x11, x12, x21, x22 *mat.CMatrix, theta, phi *mat.Vector, taup1, taup2, tauq1, tauq2, work *mat.CVector, lwork int) (err error) {
	var colmajor, lquery bool
	var one complex128
	var realone, z1, z2, z3, z4 float64
	var i, lworkmin, lworkopt int

	realone = 1.0
	one = (1.0 + 0.0*1i)

	//     Test input arguments
	colmajor = trans != Trans
	if signs != 'O' {
		z1 = realone
		z2 = realone
		z3 = realone
		z4 = realone
	} else {
		z1 = realone
		z2 = -realone
		z3 = realone
		z4 = -realone
	}
	lquery = lwork == -1

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if p < 0 || p > m {
		err = fmt.Errorf("p < 0 || p > m: p=%v, m=%v", p, m)
	} else if q < 0 || q > p || q > m-p || q > m-q {
		err = fmt.Errorf("q < 0 || q > p || q > m-p || q > m-q: q=%v, p=%v, m=%v", q, p, m)
	} else if colmajor && x11.Rows < max(1, p) {
		err = fmt.Errorf("colmajor && x11.Rows < max(1, p): colmajor=%v, x11.Rows=%v, p=%v", colmajor, x11.Rows, p)
	} else if !colmajor && x11.Rows < max(1, q) {
		err = fmt.Errorf("!colmajor && x11.Rows < max(1, q): colmajor=%v, x11.Rows=%v, q=%v", colmajor, x11.Rows, q)
	} else if colmajor && x12.Rows < max(1, p) {
		err = fmt.Errorf("colmajor && x12.Rows < max(1, p): colmajor=%v, x12.Rows=%v, p=%v", colmajor, x12.Rows, p)
	} else if !colmajor && x12.Rows < max(1, m-q) {
		err = fmt.Errorf("!colmajor && x12.Rows < max(1, m-q): colmajor=%v, x12.Major=%v, m=%v, q=%v", colmajor, x12.Rows, m, q)
	} else if colmajor && x21.Rows < max(1, m-p) {
		err = fmt.Errorf("colmajor && x21.Rows < max(1, m-p): colmajor=%v, x12.Rows=%v, m=%v, p=%v", colmajor, x12.Rows, m, p)
	} else if !colmajor && x21.Rows < max(1, q) {
		err = fmt.Errorf("!colmajor && x21.Rows < max(1, q): colmajor=%v, x21.Rows=%v, q=%v", colmajor, x21.Rows, q)
	} else if colmajor && x22.Rows < max(1, m-p) {
		err = fmt.Errorf("colmajor && x22.Rows < max(1, m-p): colmajor=%v, x22.Rows=%v, m=%v, p=%v", colmajor, x22.Rows, m, p)
	} else if !colmajor && x22.Rows < max(1, m-q) {
		err = fmt.Errorf("!colmajor && x22.Rows < max(1, m-q): colmajor=%v, x22.Rows=%v, m=%v, q=%v", colmajor, x22.Rows, m, q)
	}

	//     Compute workspace
	if err == nil {
		lworkopt = m - q
		lworkmin = m - q
		work.SetRe(0, float64(lworkopt))
		if lwork < lworkmin && !lquery {
			err = fmt.Errorf("lwork < lworkmin && !lquery: lwork=%v, lworkmin=%v, lquery=%v", lwork, lworkmin, lquery)
		}
	}
	if err != nil {
		gltest.Xerbla2("Zurbdb", err)
		return
	} else if lquery {
		return
	}

	//     Handle column-major and row-major separately
	if colmajor {
		//        Reduce columns 1, ..., Q of X11, X12, X21, and X22
		for i = 1; i <= q; i++ {

			if i == 1 {
				goblas.Zscal(p-i+1, complex(z1, 0), x11.CVector(i-1, i-1, 1))
			} else {
				goblas.Zscal(p-i+1, complex(z1*math.Cos(phi.Get(i-1-1)), 0), x11.CVector(i-1, i-1, 1))
				goblas.Zaxpy(p-i+1, complex(-z1*z3*z4*math.Sin(phi.Get(i-1-1)), 0), x12.CVector(i-1, i-1-1, 1), x11.CVector(i-1, i-1, 1))
			}
			if i == 1 {
				goblas.Zscal(m-p-i+1, complex(z2, 0), x21.CVector(i-1, i-1, 1))
			} else {
				goblas.Zscal(m-p-i+1, complex(z2*math.Cos(phi.Get(i-1-1)), 0), x21.CVector(i-1, i-1, 1))
				goblas.Zaxpy(m-p-i+1, complex(-z2*z3*z4*math.Sin(phi.Get(i-1-1)), 0), x22.CVector(i-1, i-1-1, 1), x21.CVector(i-1, i-1, 1))
			}

			theta.Set(i-1, math.Atan2(goblas.Dznrm2(m-p-i+1, x21.CVector(i-1, i-1, 1)), goblas.Dznrm2(p-i+1, x11.CVector(i-1, i-1, 1))))

			if p > i {
				*x11.GetPtr(i-1, i-1), *taup1.GetPtr(i - 1) = Zlarfgp(p-i+1, x11.Get(i-1, i-1), x11.CVector(i, i-1, 1))
			} else if p == i {
				*x11.GetPtr(i-1, i-1), *taup1.GetPtr(i - 1) = Zlarfgp(p-i+1, x11.Get(i-1, i-1), x11.CVector(i-1, i-1, 1))
			}
			x11.Set(i-1, i-1, one)
			if m-p > i {
				*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Zlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.CVector(i, i-1, 1))
			} else if m-p == i {
				*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Zlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.CVector(i-1, i-1, 1))
			}
			x21.Set(i-1, i-1, one)

			if q > i {
				Zlarf(Left, p-i+1, q-i, x11.CVector(i-1, i-1, 1), taup1.GetConj(i-1), x11.Off(i-1, i), work)
				Zlarf(Left, m-p-i+1, q-i, x21.CVector(i-1, i-1, 1), taup2.GetConj(i-1), x21.Off(i-1, i), work)
			}
			if m-q+1 > i {
				Zlarf(Left, p-i+1, m-q-i+1, x11.CVector(i-1, i-1, 1), taup1.GetConj(i-1), x12.Off(i-1, i-1), work)
				Zlarf(Left, m-p-i+1, m-q-i+1, x21.CVector(i-1, i-1, 1), taup2.GetConj(i-1), x22.Off(i-1, i-1), work)
			}

			if i < q {
				goblas.Zscal(q-i, complex(-z1*z3*math.Sin(theta.Get(i-1)), 0), x11.CVector(i-1, i))
				goblas.Zaxpy(q-i, complex(z2*z3*math.Cos(theta.Get(i-1)), 0), x21.CVector(i-1, i), x11.CVector(i-1, i))
			}
			goblas.Zscal(m-q-i+1, complex(-z1*z4*math.Sin(theta.Get(i-1)), 0), x12.CVector(i-1, i-1))
			goblas.Zaxpy(m-q-i+1, complex(z2*z4*math.Cos(theta.Get(i-1)), 0), x22.CVector(i-1, i-1), x12.CVector(i-1, i-1))

			if i < q {
				phi.Set(i-1, math.Atan2(goblas.Dznrm2(q-i, x11.CVector(i-1, i)), goblas.Dznrm2(m-q-i+1, x12.CVector(i-1, i-1))))
			}

			if i < q {
				Zlacgv(q-i, x11.CVector(i-1, i))
				if i == q-1 {
					*x11.GetPtr(i-1, i), *tauq1.GetPtr(i - 1) = Zlarfgp(q-i, x11.Get(i-1, i), x11.CVector(i-1, i))
				} else {
					*x11.GetPtr(i-1, i), *tauq1.GetPtr(i - 1) = Zlarfgp(q-i, x11.Get(i-1, i), x11.CVector(i-1, i+2-1))
				}
				x11.Set(i-1, i, one)
			}
			if m-q+1 > i {
				Zlacgv(m-q-i+1, x12.CVector(i-1, i-1))
				if m-q == i {
					*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Zlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.CVector(i-1, i-1))
				} else {
					*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Zlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.CVector(i-1, i))
				}
			}
			x12.Set(i-1, i-1, one)

			if i < q {
				Zlarf(Right, p-i, q-i, x11.CVector(i-1, i), tauq1.Get(i-1), x11.Off(i, i), work)
				Zlarf(Right, m-p-i, q-i, x11.CVector(i-1, i), tauq1.Get(i-1), x21.Off(i, i), work)
			}
			if p > i {
				Zlarf(Right, p-i, m-q-i+1, x12.CVector(i-1, i-1), tauq2.Get(i-1), x12.Off(i, i-1), work)
			}
			if m-p > i {
				Zlarf(Right, m-p-i, m-q-i+1, x12.CVector(i-1, i-1), tauq2.Get(i-1), x22.Off(i, i-1), work)
			}

			if i < q {
				Zlacgv(q-i, x11.CVector(i-1, i))
			}
			Zlacgv(m-q-i+1, x12.CVector(i-1, i-1))

		}

		//        Reduce columns Q + 1, ..., P of X12, X22
		for i = q + 1; i <= p; i++ {

			goblas.Zscal(m-q-i+1, complex(-z1*z4, 0), x12.CVector(i-1, i-1))
			Zlacgv(m-q-i+1, x12.CVector(i-1, i-1))
			if i >= m-q {
				*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Zlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.CVector(i-1, i-1))
			} else {
				*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Zlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.CVector(i-1, i))
			}
			x12.Set(i-1, i-1, one)

			if p > i {
				Zlarf(Right, p-i, m-q-i+1, x12.CVector(i-1, i-1), tauq2.Get(i-1), x12.Off(i, i-1), work)
			}
			if m-p-q >= 1 {
				Zlarf(Right, m-p-q, m-q-i+1, x12.CVector(i-1, i-1), tauq2.Get(i-1), x22.Off(q, i-1), work)
			}

			Zlacgv(m-q-i+1, x12.CVector(i-1, i-1))

		}

		//        Reduce columns P + 1, ..., M - Q of X12, X22
		for i = 1; i <= m-p-q; i++ {

			goblas.Zscal(m-p-q-i+1, complex(z2*z4, 0), x22.CVector(q+i-1, p+i-1))
			Zlacgv(m-p-q-i+1, x22.CVector(q+i-1, p+i-1))
			*x22.GetPtr(q+i-1, p+i-1), *tauq2.GetPtr(p + i - 1) = Zlarfgp(m-p-q-i+1, x22.Get(q+i-1, p+i-1), x22.CVector(q+i-1, p+i))
			x22.Set(q+i-1, p+i-1, one)
			Zlarf(Right, m-p-q-i, m-p-q-i+1, x22.CVector(q+i-1, p+i-1), tauq2.Get(p+i-1), x22.Off(q+i, p+i-1), work)

			Zlacgv(m-p-q-i+1, x22.CVector(q+i-1, p+i-1))

		}

	} else {
		//        Reduce columns 1, ..., Q of X11, X12, X21, X22
		for i = 1; i <= q; i++ {

			if i == 1 {
				goblas.Zscal(p-i+1, complex(z1, 0), x11.CVector(i-1, i-1))
			} else {
				goblas.Zscal(p-i+1, complex(z1*math.Cos(phi.Get(i-1-1)), 0), x11.CVector(i-1, i-1))
				goblas.Zaxpy(p-i+1, complex(-z1*z3*z4*math.Sin(phi.Get(i-1-1)), 0), x12.CVector(i-1-1, i-1), x11.CVector(i-1, i-1))
			}
			if i == 1 {
				goblas.Zscal(m-p-i+1, complex(z2, 0), x21.CVector(i-1, i-1))
			} else {
				goblas.Zscal(m-p-i+1, complex(z2*math.Cos(phi.Get(i-1-1)), 0), x21.CVector(i-1, i-1))
				goblas.Zaxpy(m-p-i+1, complex(-z2*z3*z4*math.Sin(phi.Get(i-1-1)), 0), x22.CVector(i-1-1, i-1), x21.CVector(i-1, i-1))
			}

			theta.Set(i-1, math.Atan2(goblas.Dznrm2(m-p-i+1, x21.CVector(i-1, i-1)), goblas.Dznrm2(p-i+1, x11.CVector(i-1, i-1))))

			Zlacgv(p-i+1, x11.CVector(i-1, i-1))
			Zlacgv(m-p-i+1, x21.CVector(i-1, i-1))

			*x11.GetPtr(i-1, i-1), *taup1.GetPtr(i - 1) = Zlarfgp(p-i+1, x11.Get(i-1, i-1), x11.CVector(i-1, i))
			x11.Set(i-1, i-1, one)
			if i == m-p {
				*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Zlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.CVector(i-1, i-1))
			} else {
				*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Zlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.CVector(i-1, i))
			}
			x21.Set(i-1, i-1, one)

			Zlarf(Right, q-i, p-i+1, x11.CVector(i-1, i-1), taup1.Get(i-1), x11.Off(i, i-1), work)
			Zlarf(Right, m-q-i+1, p-i+1, x11.CVector(i-1, i-1), taup1.Get(i-1), x12.Off(i-1, i-1), work)
			Zlarf(Right, q-i, m-p-i+1, x21.CVector(i-1, i-1), taup2.Get(i-1), x21.Off(i, i-1), work)
			Zlarf(Right, m-q-i+1, m-p-i+1, x21.CVector(i-1, i-1), taup2.Get(i-1), x22.Off(i-1, i-1), work)

			Zlacgv(p-i+1, x11.CVector(i-1, i-1))
			Zlacgv(m-p-i+1, x21.CVector(i-1, i-1))

			if i < q {
				goblas.Zscal(q-i, complex(-z1*z3*math.Sin(theta.Get(i-1)), 0), x11.CVector(i, i-1, 1))
				goblas.Zaxpy(q-i, complex(z2*z3*math.Cos(theta.Get(i-1)), 0), x21.CVector(i, i-1, 1), x11.CVector(i, i-1, 1))
			}
			goblas.Zscal(m-q-i+1, complex(-z1*z4*math.Sin(theta.Get(i-1)), 0), x12.CVector(i-1, i-1, 1))
			goblas.Zaxpy(m-q-i+1, complex(z2*z4*math.Cos(theta.Get(i-1)), 0), x22.CVector(i-1, i-1, 1), x12.CVector(i-1, i-1, 1))

			if i < q {
				phi.Set(i-1, math.Atan2(goblas.Dznrm2(q-i, x11.CVector(i, i-1, 1)), goblas.Dznrm2(m-q-i+1, x12.CVector(i-1, i-1, 1))))
			}

			if i < q {
				*x11.GetPtr(i, i-1), *tauq1.GetPtr(i - 1) = Zlarfgp(q-i, x11.Get(i, i-1), x11.CVector(i+2-1, i-1, 1))
				x11.Set(i, i-1, one)
			}
			*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Zlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.CVector(i, i-1, 1))
			x12.Set(i-1, i-1, one)

			if i < q {
				Zlarf(Left, q-i, p-i, x11.CVector(i, i-1, 1), tauq1.GetConj(i-1), x11.Off(i, i), work)
				Zlarf(Left, q-i, m-p-i, x11.CVector(i, i-1, 1), tauq1.GetConj(i-1), x21.Off(i, i), work)
			}
			Zlarf(Left, m-q-i+1, p-i, x12.CVector(i-1, i-1, 1), tauq2.GetConj(i-1), x12.Off(i-1, i), work)
			if m-p > i {
				Zlarf(Left, m-q-i+1, m-p-i, x12.CVector(i-1, i-1, 1), tauq2.GetConj(i-1), x22.Off(i-1, i), work)
			}

		}

		//        Reduce columns Q + 1, ..., P of X12, X22
		for i = q + 1; i <= p; i++ {

			goblas.Zscal(m-q-i+1, complex(-z1*z4, 0), x12.CVector(i-1, i-1, 1))
			*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Zlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.CVector(i, i-1, 1))
			x12.Set(i-1, i-1, one)

			if p > i {
				Zlarf(Left, m-q-i+1, p-i, x12.CVector(i-1, i-1, 1), tauq2.GetConj(i-1), x12.Off(i-1, i), work)
			}
			if m-p-q >= 1 {
				Zlarf(Left, m-q-i+1, m-p-q, x12.CVector(i-1, i-1, 1), tauq2.GetConj(i-1), x22.Off(i-1, q), work)
			}

		}

		//        Reduce columns P + 1, ..., M - Q of X12, X22
		for i = 1; i <= m-p-q; i++ {

			goblas.Zscal(m-p-q-i+1, complex(z2*z4, 0), x22.CVector(p+i-1, q+i-1, 1))
			*x22.GetPtr(p+i-1, q+i-1), *tauq2.GetPtr(p + i - 1) = Zlarfgp(m-p-q-i+1, x22.Get(p+i-1, q+i-1), x22.CVector(p+i, q+i-1, 1))
			x22.Set(p+i-1, q+i-1, one)

			if m-p-q != i {
				Zlarf(Left, m-p-q-i+1, m-p-q-i, x22.CVector(p+i-1, q+i-1, 1), tauq2.GetConj(p+i-1), x22.Off(p+i-1, q+i), work)
			}

		}

	}

	return
}
