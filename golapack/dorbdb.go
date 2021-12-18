package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorbdb simultaneously bidiagonalizes the blocks of an M-by-M
// partitioned orthogonal matrix X:
//
//                                 [ B11 | B12 0  0 ]
//     [ X11 | X12 ]   [ P1 |    ] [  0  |  0 -I  0 ] [ Q1 |    ]**T
// X = [-----------] = [---------] [----------------] [---------]   .
//     [ X21 | X22 ]   [    | P2 ] [ B21 | B22 0  0 ] [    | Q2 ]
//                                 [  0  |  0  0  I ]
//
// X11 is P-by-Q. Q must be no larger than P, M-P, or M-Q. (If this is
// not the case, then X must be transposed and/or permuted. This can be
// done in constant time using the TRANS and SIGNS options. See DORCSD
// for details.)
//
// The orthogonal matrices P1, P2, Q1, and Q2 are P-by-P, (M-P)-by-
// (M-P), Q-by-Q, and (M-Q)-by-(M-Q), respectively. They are
// represented implicitly by Householder vectors.
//
// B11, B12, B21, and B22 are Q-by-Q bidiagonal matrices represented
// implicitly by angles THETA, PHI.
func Dorbdb(trans mat.MatTrans, signs byte, m, p, q int, x11, x12, x21, x22 *mat.Matrix, theta, phi, taup1, taup2, tauq1, tauq2, work *mat.Vector, lwork int) (err error) {
	var colmajor, lquery bool
	var one, realone, z1, z2, z3, z4 float64
	var i, lworkmin, lworkopt int

	realone = 1.0
	one = 1.0

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
		err = fmt.Errorf("colmajor && x11.Rows < max(1, p): trans=%s, x11.Rows=%v, p=%v", trans, x11.Rows, p)
	} else if !colmajor && x11.Rows < max(1, q) {
		err = fmt.Errorf("!colmajor && x11.Rows < max(1, q): trans=%s, x11.Rows=%v, q=%v", trans, x11.Rows, q)
	} else if colmajor && x12.Rows < max(1, p) {
		err = fmt.Errorf("colmajor && x12.Rows < max(1, p): trans=%s, x12.Rows=%v, p=%v", trans, x12.Rows, p)
	} else if !colmajor && x12.Rows < max(1, m-q) {
		err = fmt.Errorf("!colmajor && x12.Rows < max(1, m-q): trans=%s, x12.Rows=%v, m=%v, q=%v", trans, x12.Rows, m, q)
	} else if colmajor && x21.Rows < max(1, m-p) {
		err = fmt.Errorf("colmajor && x12.Rows < max(1, m-p): trans=%s, x21.Rows=%v, m=%v, p=%v", trans, x21.Rows, m, p)
	} else if !colmajor && x21.Rows < max(1, q) {
		err = fmt.Errorf("!colmajor && x21.Rows < max(1, q): trans=%s, x21.Rows=%v, q=%v", trans, x21.Rows, q)
	} else if colmajor && x22.Rows < max(1, m-p) {
		err = fmt.Errorf("colmajor && x22.Rows < max(1, m-p): trans=%s, x22.Rows=%v, m=%v, p=%v", trans, x22.Rows, m, p)
	} else if !colmajor && x22.Rows < max(1, m-q) {
		err = fmt.Errorf("!colmajor && x22.Rows < max(1, m-q): trans=%s, x22.Rows=%v, m=%v, q=%v", trans, x22.Rows, m, q)
	}

	//     Compute workspace
	if err == nil {
		lworkopt = m - q
		lworkmin = m - q
		work.Set(0, float64(lworkopt))
		if lwork < lworkmin && !lquery {
			err = fmt.Errorf("lwork < lworkmin && !lquery: lwork=%v, lworkmin=%v, lquery=%v", lwork, lworkmin, lquery)
		}
	}
	if err != nil {
		gltest.Xerbla2("Dorbdb", err)
		return
	} else if lquery {
		return
	}

	//     Handle column-major and row-major separately
	if colmajor {
		//        Reduce columns 1, ..., Q of X11, X12, X21, and X22
		for i = 1; i <= q; i++ {

			if i == 1 {
				x11.Off(i-1, i-1).Vector().Scal(p-i+1, z1, 1)
			} else {
				x11.Off(i-1, i-1).Vector().Scal(p-i+1, z1*math.Cos(phi.Get(i-1-1)), 1)
				x11.Off(i-1, i-1).Vector().Axpy(p-i+1, -z1*z3*z4*math.Sin(phi.Get(i-1-1)), x12.Off(i-1, i-1-1).Vector(), 1, 1)
			}
			if i == 1 {
				x21.Off(i-1, i-1).Vector().Scal(m-p-i+1, z2, 1)
			} else {
				x21.Off(i-1, i-1).Vector().Scal(m-p-i+1, z2*math.Cos(phi.Get(i-1-1)), 1)
				x21.Off(i-1, i-1).Vector().Axpy(m-p-i+1, -z2*z3*z4*math.Sin(phi.Get(i-1-1)), x22.Off(i-1, i-1-1).Vector(), 1, 1)
			}

			theta.Set(i-1, math.Atan2(x21.Off(i-1, i-1).Vector().Nrm2(m-p-i+1, 1), x11.Off(i-1, i-1).Vector().Nrm2(p-i+1, 1)))

			if p > i {
				*x11.GetPtr(i-1, i-1), *taup1.GetPtr(i - 1) = Dlarfgp(p-i+1, x11.Get(i-1, i-1), x11.Off(i, i-1).Vector(), 1)
			} else if p == i {
				*x11.GetPtr(i-1, i-1), *taup1.GetPtr(i - 1) = Dlarfgp(p-i+1, x11.Get(i-1, i-1), x11.Off(i-1, i-1).Vector(), 1)
			}
			x11.Set(i-1, i-1, one)
			if m-p > i {
				*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Dlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.Off(i, i-1).Vector(), 1)
			} else if m-p == i {
				*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Dlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.Off(i-1, i-1).Vector(), 1)
			}
			x21.Set(i-1, i-1, one)

			if q > i {
				Dlarf(Left, p-i+1, q-i, x11.Off(i-1, i-1).Vector(), 1, taup1.Get(i-1), x11.Off(i-1, i), work)
			}
			if m-q+1 > i {
				Dlarf(Left, p-i+1, m-q-i+1, x11.Off(i-1, i-1).Vector(), 1, taup1.Get(i-1), x12.Off(i-1, i-1), work)
			}
			if q > i {
				Dlarf(Left, m-p-i+1, q-i, x21.Off(i-1, i-1).Vector(), 1, taup2.Get(i-1), x21.Off(i-1, i), work)
			}
			if m-q+1 > i {
				Dlarf(Left, m-p-i+1, m-q-i+1, x21.Off(i-1, i-1).Vector(), 1, taup2.Get(i-1), x22.Off(i-1, i-1), work)
			}

			if i < q {
				x11.Off(i-1, i).Vector().Scal(q-i, -z1*z3*math.Sin(theta.Get(i-1)), x11.Rows)
				x11.Off(i-1, i).Vector().Axpy(q-i, z2*z3*math.Cos(theta.Get(i-1)), x21.Off(i-1, i).Vector(), x21.Rows, x11.Rows)
			}
			x12.Off(i-1, i-1).Vector().Scal(m-q-i+1, -z1*z4*math.Sin(theta.Get(i-1)), x12.Rows)
			x12.Off(i-1, i-1).Vector().Axpy(m-q-i+1, z2*z4*math.Cos(theta.Get(i-1)), x22.Off(i-1, i-1).Vector(), x22.Rows, x12.Rows)
			//
			if i < q {
				phi.Set(i-1, math.Atan2(x11.Off(i-1, i).Vector().Nrm2(q-i, x11.Rows), x12.Off(i-1, i-1).Vector().Nrm2(m-q-i+1, x12.Rows)))
			}

			if i < q {
				if q-i == 1 {
					*x11.GetPtr(i-1, i), *tauq1.GetPtr(i - 1) = Dlarfgp(q-i, x11.Get(i-1, i), x11.Off(i-1, i).Vector(), x11.Rows)
				} else {
					*x11.GetPtr(i-1, i), *tauq1.GetPtr(i - 1) = Dlarfgp(q-i, x11.Get(i-1, i), x11.Off(i-1, i+2-1).Vector(), x11.Rows)
				}
				x11.Set(i-1, i, one)
			}
			if q+i-1 < m {
				if m-q == i {
					*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Dlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.Off(i-1, i-1).Vector(), x12.Rows)
				} else {
					*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Dlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.Off(i-1, i).Vector(), x12.Rows)
				}
			}
			x12.Set(i-1, i-1, one)
			//
			if i < q {
				Dlarf(Right, p-i, q-i, x11.Off(i-1, i).Vector(), x11.Rows, tauq1.Get(i-1), x11.Off(i, i), work)
				Dlarf(Right, m-p-i, q-i, x11.Off(i-1, i).Vector(), x11.Rows, tauq1.Get(i-1), x21.Off(i, i), work)
			}
			if p > i {
				Dlarf(Right, p-i, m-q-i+1, x12.Off(i-1, i-1).Vector(), x12.Rows, tauq2.Get(i-1), x12.Off(i, i-1), work)
			}
			if m-p > i {
				Dlarf(Right, m-p-i, m-q-i+1, x12.Off(i-1, i-1).Vector(), x12.Rows, tauq2.Get(i-1), x22.Off(i, i-1), work)
			}

		}

		//        Reduce columns Q + 1, ..., P of X12, X22
		for i = q + 1; i <= p; i++ {

			x12.Off(i-1, i-1).Vector().Scal(m-q-i+1, -z1*z4, x12.Rows)
			if i >= m-q {
				*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Dlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.Off(i-1, i-1).Vector(), x12.Rows)
			} else {
				*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Dlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.Off(i-1, i).Vector(), x12.Rows)
			}
			x12.Set(i-1, i-1, one)

			if p > i {
				Dlarf(Right, p-i, m-q-i+1, x12.Off(i-1, i-1).Vector(), x12.Rows, tauq2.Get(i-1), x12.Off(i, i-1), work)
			}
			if m-p-q >= 1 {
				Dlarf(Right, m-p-q, m-q-i+1, x12.Off(i-1, i-1).Vector(), x12.Rows, tauq2.Get(i-1), x22.Off(q, i-1), work)
			}

		}

		//        Reduce columns P + 1, ..., M - Q of X12, X22
		for i = 1; i <= m-p-q; i++ {

			x22.Off(q+i-1, p+i-1).Vector().Scal(m-p-q-i+1, z2*z4, x22.Rows)
			if i == m-p-q {
				*x22.GetPtr(q+i-1, p+i-1), *tauq2.GetPtr(p + i - 1) = Dlarfgp(m-p-q-i+1, x22.Get(q+i-1, p+i-1), x22.Off(q+i-1, p+i-1).Vector(), x22.Rows)
			} else {
				*x22.GetPtr(q+i-1, p+i-1), *tauq2.GetPtr(p + i - 1) = Dlarfgp(m-p-q-i+1, x22.Get(q+i-1, p+i-1), x22.Off(q+i-1, p+i).Vector(), x22.Rows)
			}
			x22.Set(q+i-1, p+i-1, one)
			if i < m-p-q {
				Dlarf(Right, m-p-q-i, m-p-q-i+1, x22.Off(q+i-1, p+i-1).Vector(), x22.Rows, tauq2.Get(p+i-1), x22.Off(q+i, p+i-1), work)
			}

		}

	} else {
		//        Reduce columns 1, ..., Q of X11, X12, X21, X22
		for i = 1; i <= q; i++ {

			if i == 1 {
				x11.Off(i-1, i-1).Vector().Scal(p-i+1, z1, x11.Rows)
			} else {
				x11.Off(i-1, i-1).Vector().Scal(p-i+1, z1*math.Cos(phi.Get(i-1-1)), x11.Rows)
				x11.Off(i-1, i-1).Vector().Axpy(p-i+1, -z1*z3*z4*math.Sin(phi.Get(i-1-1)), x12.Off(i-1-1, i-1).Vector(), x12.Rows, x11.Rows)
			}
			if i == 1 {
				x21.Off(i-1, i-1).Vector().Scal(m-p-i+1, z2, x21.Rows)
			} else {
				x21.Off(i-1, i-1).Vector().Scal(m-p-i+1, z2*math.Cos(phi.Get(i-1-1)), x21.Rows)
				x21.Off(i-1, i-1).Vector().Axpy(m-p-i+1, -z2*z3*z4*math.Sin(phi.Get(i-1-1)), x22.Off(i-1-1, i-1).Vector(), x22.Rows, x21.Rows)
			}

			theta.Set(i-1, math.Atan2(x21.Off(i-1, i-1).Vector().Nrm2(m-p-i+1, x21.Rows), x11.Off(i-1, i-1).Vector().Nrm2(p-i+1, x11.Rows)))

			*x11.GetPtr(i-1, i-1), *taup1.GetPtr(i - 1) = Dlarfgp(p-i+1, x11.Get(i-1, i-1), x11.Off(i-1, i).Vector(), x11.Rows)
			x11.Set(i-1, i-1, one)
			if i == m-p {
				*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Dlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.Off(i-1, i-1).Vector(), x21.Rows)
			} else {
				*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Dlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.Off(i-1, i).Vector(), x21.Rows)
			}
			x21.Set(i-1, i-1, one)

			if q > i {
				Dlarf(Right, q-i, p-i+1, x11.Off(i-1, i-1).Vector(), x11.Rows, taup1.Get(i-1), x11.Off(i, i-1), work)
			}
			if m-q+1 > i {
				Dlarf(Right, m-q-i+1, p-i+1, x11.Off(i-1, i-1).Vector(), x11.Rows, taup1.Get(i-1), x12.Off(i-1, i-1), work)
			}
			if q > i {
				Dlarf(Right, q-i, m-p-i+1, x21.Off(i-1, i-1).Vector(), x21.Rows, taup2.Get(i-1), x21.Off(i, i-1), work)
			}
			if m-q+1 > i {
				Dlarf(Right, m-q-i+1, m-p-i+1, x21.Off(i-1, i-1).Vector(), x21.Rows, taup2.Get(i-1), x22.Off(i-1, i-1), work)
			}

			if i < q {
				x11.Off(i, i-1).Vector().Scal(q-i, -z1*z3*math.Sin(theta.Get(i-1)), 1)
				x11.Off(i, i-1).Vector().Axpy(q-i, z2*z3*math.Cos(theta.Get(i-1)), x21.Off(i, i-1).Vector(), 1, 1)
			}
			x12.Off(i-1, i-1).Vector().Scal(m-q-i+1, -z1*z4*math.Sin(theta.Get(i-1)), 1)
			x12.Off(i-1, i-1).Vector().Axpy(m-q-i+1, z2*z4*math.Cos(theta.Get(i-1)), x22.Off(i-1, i-1).Vector(), 1, 1)

			if i < q {
				phi.Set(i-1, math.Atan2(x11.Off(i, i-1).Vector().Nrm2(q-i, 1), x12.Off(i-1, i-1).Vector().Nrm2(m-q-i+1, 1)))
			}

			if i < q {
				if q-i == 1 {
					*x11.GetPtr(i, i-1), *tauq1.GetPtr(i - 1) = Dlarfgp(q-i, x11.Get(i, i-1), x11.Off(i, i-1).Vector(), 1)
				} else {
					*x11.GetPtr(i, i-1), *tauq1.GetPtr(i - 1) = Dlarfgp(q-i, x11.Get(i, i-1), x11.Off(i+2-1, i-1).Vector(), 1)
				}
				x11.Set(i, i-1, one)
			}
			if m-q > i {
				*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Dlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.Off(i, i-1).Vector(), 1)
			} else {
				*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Dlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.Off(i-1, i-1).Vector(), 1)
			}
			x12.Set(i-1, i-1, one)

			if i < q {
				Dlarf(Left, q-i, p-i, x11.Off(i, i-1).Vector(), 1, tauq1.Get(i-1), x11.Off(i, i), work)
				Dlarf(Left, q-i, m-p-i, x11.Off(i, i-1).Vector(), 1, tauq1.Get(i-1), x21.Off(i, i), work)
			}
			Dlarf(Left, m-q-i+1, p-i, x12.Off(i-1, i-1).Vector(), 1, tauq2.Get(i-1), x12.Off(i-1, i), work)
			if m-p-i > 0 {
				Dlarf(Left, m-q-i+1, m-p-i, x12.Off(i-1, i-1).Vector(), 1, tauq2.Get(i-1), x22.Off(i-1, i), work)
			}

		}

		//        Reduce columns Q + 1, ..., P of X12, X22
		for i = q + 1; i <= p; i++ {

			x12.Off(i-1, i-1).Vector().Scal(m-q-i+1, -z1*z4, 1)
			*x12.GetPtr(i-1, i-1), *tauq2.GetPtr(i - 1) = Dlarfgp(m-q-i+1, x12.Get(i-1, i-1), x12.Off(i, i-1).Vector(), 1)
			x12.Set(i-1, i-1, one)

			if p > i {
				Dlarf(Left, m-q-i+1, p-i, x12.Off(i-1, i-1).Vector(), 1, tauq2.Get(i-1), x12.Off(i-1, i), work)
			}
			if m-p-q >= 1 {
				Dlarf(Left, m-q-i+1, m-p-q, x12.Off(i-1, i-1).Vector(), 1, tauq2.Get(i-1), x22.Off(i-1, q), work)
			}

		}

		//        Reduce columns P + 1, ..., M - Q of X12, X22
		for i = 1; i <= m-p-q; i++ {

			x22.Off(p+i-1, q+i-1).Vector().Scal(m-p-q-i+1, z2*z4, 1)
			if m-p-q == i {
				*x22.GetPtr(p+i-1, q+i-1), *tauq2.GetPtr(p + i - 1) = Dlarfgp(m-p-q-i+1, x22.Get(p+i-1, q+i-1), x22.Off(p+i-1, q+i-1).Vector(), 1)
			} else {
				*x22.GetPtr(p+i-1, q+i-1), *tauq2.GetPtr(p + i - 1) = Dlarfgp(m-p-q-i+1, x22.Get(p+i-1, q+i-1), x22.Off(p+i, q+i-1).Vector(), 1)
				Dlarf(Left, m-p-q-i+1, m-p-q-i, x22.Off(p+i-1, q+i-1).Vector(), 1, tauq2.Get(p+i-1), x22.Off(p+i-1, q+i), work)
			}
			x22.Set(p+i-1, q+i-1, one)

		}

	}

	return
}
