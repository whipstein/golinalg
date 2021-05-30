package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
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
func Dorbdb(trans, signs byte, m, p, q *int, x11 *mat.Matrix, ldx11 *int, x12 *mat.Matrix, ldx12 *int, x21 *mat.Matrix, ldx21 *int, x22 *mat.Matrix, ldx22 *int, theta, phi, taup1, taup2, tauq1, tauq2, work *mat.Vector, lwork, info *int) {
	var colmajor, lquery bool
	var one, realone, z1, z2, z3, z4 float64
	var i, lworkmin, lworkopt int

	realone = 1.0
	one = 1.0

	//     Test input arguments
	(*info) = 0
	colmajor = trans != 'T'
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
	lquery = (*lwork) == -1

	if (*m) < 0 {
		(*info) = -3
	} else if (*p) < 0 || (*p) > (*m) {
		(*info) = -4
	} else if (*q) < 0 || (*q) > (*p) || (*q) > (*m)-(*p) || (*q) > (*m)-(*q) {
		(*info) = -5
	} else if colmajor && (*ldx11) < maxint(1, *p) {
		(*info) = -7
	} else if !colmajor && (*ldx11) < maxint(1, *q) {
		(*info) = -7
	} else if colmajor && (*ldx12) < maxint(1, *p) {
		(*info) = -9
	} else if !colmajor && (*ldx12) < maxint(1, (*m)-(*q)) {
		(*info) = -9
	} else if colmajor && (*ldx21) < maxint(1, (*m)-(*p)) {
		(*info) = -11
	} else if !colmajor && (*ldx21) < maxint(1, *q) {
		(*info) = -11
	} else if colmajor && (*ldx22) < maxint(1, (*m)-(*p)) {
		(*info) = -13
	} else if !colmajor && (*ldx22) < maxint(1, (*m)-(*q)) {
		(*info) = -13
	}

	//     Compute workspace
	if (*info) == 0 {
		lworkopt = (*m) - (*q)
		lworkmin = (*m) - (*q)
		work.Set(0, float64(lworkopt))
		if (*lwork) < lworkmin && !lquery {
			(*info) = -21
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("xORBDB"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Handle column-major and row-major separately
	if colmajor {
		//        Reduce columns 1, ..., Q of X11, X12, X21, and X22
		for i = 1; i <= (*q); i++ {

			if i == 1 {
				goblas.Dscal(toPtr((*p)-i+1), &z1, x11.Vector(i-1, i-1), func() *int { y := 1; return &y }())
			} else {
				goblas.Dscal(toPtr((*p)-i+1), toPtrf64(z1*math.Cos(phi.Get(i-1-1))), x11.Vector(i-1, i-1), func() *int { y := 1; return &y }())
				goblas.Daxpy(toPtr((*p)-i+1), toPtrf64(-z1*z3*z4*math.Sin(phi.Get(i-1-1))), x12.Vector(i-1, i-1-1), func() *int { y := 1; return &y }(), x11.Vector(i-1, i-1), func() *int { y := 1; return &y }())
			}
			if i == 1 {
				goblas.Dscal(toPtr((*m)-(*p)-i+1), &z2, x21.Vector(i-1, i-1), func() *int { y := 1; return &y }())
			} else {
				goblas.Dscal(toPtr((*m)-(*p)-i+1), toPtrf64(z2*math.Cos(phi.Get(i-1-1))), x21.Vector(i-1, i-1), func() *int { y := 1; return &y }())
				goblas.Daxpy(toPtr((*m)-(*p)-i+1), toPtrf64(-z2*z3*z4*math.Sin(phi.Get(i-1-1))), x22.Vector(i-1, i-1-1), func() *int { y := 1; return &y }(), x21.Vector(i-1, i-1), func() *int { y := 1; return &y }())
			}

			theta.Set(i-1, math.Atan2(goblas.Dnrm2(toPtr((*m)-(*p)-i+1), x21.Vector(i-1, i-1), func() *int { y := 1; return &y }()), goblas.Dnrm2(toPtr((*p)-i+1), x11.Vector(i-1, i-1), func() *int { y := 1; return &y }())))

			if (*p) > i {
				Dlarfgp(toPtr((*p)-i+1), x11.GetPtr(i-1, i-1), x11.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1))
			} else if (*p) == i {
				Dlarfgp(toPtr((*p)-i+1), x11.GetPtr(i-1, i-1), x11.Vector(i-1, i-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1))
			}
			x11.Set(i-1, i-1, one)
			if (*m)-(*p) > i {
				Dlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1), x21.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1))
			} else if (*m)-(*p) == i {
				Dlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1), x21.Vector(i-1, i-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1))
			}
			x21.Set(i-1, i-1, one)

			if (*q) > i {
				Dlarf('L', toPtr((*p)-i+1), toPtr((*q)-i), x11.Vector(i-1, i-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1), x11.Off(i-1, i+1-1), ldx11, work)
			}
			if (*m)-(*q)+1 > i {
				Dlarf('L', toPtr((*p)-i+1), toPtr((*m)-(*q)-i+1), x11.Vector(i-1, i-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1), x12.Off(i-1, i-1), ldx12, work)
			}
			if (*q) > i {
				Dlarf('L', toPtr((*m)-(*p)-i+1), toPtr((*q)-i), x21.Vector(i-1, i-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1), x21.Off(i-1, i+1-1), ldx21, work)
			}
			if (*m)-(*q)+1 > i {
				Dlarf('L', toPtr((*m)-(*p)-i+1), toPtr((*m)-(*q)-i+1), x21.Vector(i-1, i-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1), x22.Off(i-1, i-1), ldx22, work)
			}

			if i < (*q) {
				goblas.Dscal(toPtr((*q)-i), toPtrf64(-z1*z3*math.Sin(theta.Get(i-1))), x11.Vector(i-1, i+1-1), ldx11)
				goblas.Daxpy(toPtr((*q)-i), toPtrf64(z2*z3*math.Cos(theta.Get(i-1))), x21.Vector(i-1, i+1-1), ldx21, x11.Vector(i-1, i+1-1), ldx11)
			}
			goblas.Dscal(toPtr((*m)-(*q)-i+1), toPtrf64(-z1*z4*math.Sin(theta.Get(i-1))), x12.Vector(i-1, i-1), ldx12)
			goblas.Daxpy(toPtr((*m)-(*q)-i+1), toPtrf64(z2*z4*math.Cos(theta.Get(i-1))), x22.Vector(i-1, i-1), ldx22, x12.Vector(i-1, i-1), ldx12)
			//
			if i < (*q) {
				phi.Set(i-1, math.Atan2(goblas.Dnrm2(toPtr((*q)-i), x11.Vector(i-1, i+1-1), ldx11), goblas.Dnrm2(toPtr((*m)-(*q)-i+1), x12.Vector(i-1, i-1), ldx12)))
			}

			if i < (*q) {
				if (*q)-i == 1 {
					Dlarfgp(toPtr((*q)-i), x11.GetPtr(i-1, i+1-1), x11.Vector(i-1, i+1-1), ldx11, tauq1.GetPtr(i-1))
				} else {
					Dlarfgp(toPtr((*q)-i), x11.GetPtr(i-1, i+1-1), x11.Vector(i-1, i+2-1), ldx11, tauq1.GetPtr(i-1))
				}
				x11.Set(i-1, i+1-1, one)
			}
			if (*q)+i-1 < (*m) {
				if (*m)-(*q) == i {
					Dlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.Vector(i-1, i-1), ldx12, tauq2.GetPtr(i-1))
				} else {
					Dlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.Vector(i-1, i+1-1), ldx12, tauq2.GetPtr(i-1))
				}
			}
			x12.Set(i-1, i-1, one)
			//
			if i < (*q) {
				Dlarf('R', toPtr((*p)-i), toPtr((*q)-i), x11.Vector(i-1, i+1-1), ldx11, tauq1.GetPtr(i-1), x11.Off(i+1-1, i+1-1), ldx11, work)
				Dlarf('R', toPtr((*m)-(*p)-i), toPtr((*q)-i), x11.Vector(i-1, i+1-1), ldx11, tauq1.GetPtr(i-1), x21.Off(i+1-1, i+1-1), ldx21, work)
			}
			if (*p) > i {
				Dlarf('R', toPtr((*p)-i), toPtr((*m)-(*q)-i+1), x12.Vector(i-1, i-1), ldx12, tauq2.GetPtr(i-1), x12.Off(i+1-1, i-1), ldx12, work)
			}
			if (*m)-(*p) > i {
				Dlarf('R', toPtr((*m)-(*p)-i), toPtr((*m)-(*q)-i+1), x12.Vector(i-1, i-1), ldx12, tauq2.GetPtr(i-1), x22.Off(i+1-1, i-1), ldx22, work)
			}

		}

		//        Reduce columns Q + 1, ..., P of X12, X22
		for i = (*q) + 1; i <= (*p); i++ {

			goblas.Dscal(toPtr((*m)-(*q)-i+1), toPtrf64(-z1*z4), x12.Vector(i-1, i-1), ldx12)
			if i >= (*m)-(*q) {
				Dlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.Vector(i-1, i-1), ldx12, tauq2.GetPtr(i-1))
			} else {
				Dlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.Vector(i-1, i+1-1), ldx12, tauq2.GetPtr(i-1))
			}
			x12.Set(i-1, i-1, one)

			if (*p) > i {
				Dlarf('R', toPtr((*p)-i), toPtr((*m)-(*q)-i+1), x12.Vector(i-1, i-1), ldx12, tauq2.GetPtr(i-1), x12.Off(i+1-1, i-1), ldx12, work)
			}
			if (*m)-(*p)-(*q) >= 1 {
				Dlarf('R', toPtr((*m)-(*p)-(*q)), toPtr((*m)-(*q)-i+1), x12.Vector(i-1, i-1), ldx12, tauq2.GetPtr(i-1), x22.Off((*q)+1-1, i-1), ldx22, work)
			}

		}

		//        Reduce columns P + 1, ..., M - Q of X12, X22
		for i = 1; i <= (*m)-(*p)-(*q); i++ {

			goblas.Dscal(toPtr((*m)-(*p)-(*q)-i+1), toPtrf64(z2*z4), x22.Vector((*q)+i-1, (*p)+i-1), ldx22)
			if i == (*m)-(*p)-(*q) {
				Dlarfgp(toPtr((*m)-(*p)-(*q)-i+1), x22.GetPtr((*q)+i-1, (*p)+i-1), x22.Vector((*q)+i-1, (*p)+i-1), ldx22, tauq2.GetPtr((*p)+i-1))
			} else {
				Dlarfgp(toPtr((*m)-(*p)-(*q)-i+1), x22.GetPtr((*q)+i-1, (*p)+i-1), x22.Vector((*q)+i-1, (*p)+i+1-1), ldx22, tauq2.GetPtr((*p)+i-1))
			}
			x22.Set((*q)+i-1, (*p)+i-1, one)
			if i < (*m)-(*p)-(*q) {
				Dlarf('R', toPtr((*m)-(*p)-(*q)-i), toPtr((*m)-(*p)-(*q)-i+1), x22.Vector((*q)+i-1, (*p)+i-1), ldx22, tauq2.GetPtr((*p)+i-1), x22.Off((*q)+i+1-1, (*p)+i-1), ldx22, work)
			}

		}

	} else {
		//        Reduce columns 1, ..., Q of X11, X12, X21, X22
		for i = 1; i <= (*q); i++ {

			if i == 1 {
				goblas.Dscal(toPtr((*p)-i+1), &z1, x11.Vector(i-1, i-1), ldx11)
			} else {
				goblas.Dscal(toPtr((*p)-i+1), toPtrf64(z1*math.Cos(phi.Get(i-1-1))), x11.Vector(i-1, i-1), ldx11)
				goblas.Daxpy(toPtr((*p)-i+1), toPtrf64(-z1*z3*z4*math.Sin(phi.Get(i-1-1))), x12.Vector(i-1-1, i-1), ldx12, x11.Vector(i-1, i-1), ldx11)
			}
			if i == 1 {
				goblas.Dscal(toPtr((*m)-(*p)-i+1), &z2, x21.Vector(i-1, i-1), ldx21)
			} else {
				goblas.Dscal(toPtr((*m)-(*p)-i+1), toPtrf64(z2*math.Cos(phi.Get(i-1-1))), x21.Vector(i-1, i-1), ldx21)
				goblas.Daxpy(toPtr((*m)-(*p)-i+1), toPtrf64(-z2*z3*z4*math.Sin(phi.Get(i-1-1))), x22.Vector(i-1-1, i-1), ldx22, x21.Vector(i-1, i-1), ldx21)
			}

			theta.Set(i-1, math.Atan2(goblas.Dnrm2(toPtr((*m)-(*p)-i+1), x21.Vector(i-1, i-1), ldx21), goblas.Dnrm2(toPtr((*p)-i+1), x11.Vector(i-1, i-1), ldx11)))

			Dlarfgp(toPtr((*p)-i+1), x11.GetPtr(i-1, i-1), x11.Vector(i-1, i+1-1), ldx11, taup1.GetPtr(i-1))
			x11.Set(i-1, i-1, one)
			if i == (*m)-(*p) {
				Dlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1), x21.Vector(i-1, i-1), ldx21, taup2.GetPtr(i-1))
			} else {
				Dlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1), x21.Vector(i-1, i+1-1), ldx21, taup2.GetPtr(i-1))
			}
			x21.Set(i-1, i-1, one)

			if (*q) > i {
				Dlarf('R', toPtr((*q)-i), toPtr((*p)-i+1), x11.Vector(i-1, i-1), ldx11, taup1.GetPtr(i-1), x11.Off(i+1-1, i-1), ldx11, work)
			}
			if (*m)-(*q)+1 > i {
				Dlarf('R', toPtr((*m)-(*q)-i+1), toPtr((*p)-i+1), x11.Vector(i-1, i-1), ldx11, taup1.GetPtr(i-1), x12.Off(i-1, i-1), ldx12, work)
			}
			if (*q) > i {
				Dlarf('R', toPtr((*q)-i), toPtr((*m)-(*p)-i+1), x21.Vector(i-1, i-1), ldx21, taup2.GetPtr(i-1), x21.Off(i+1-1, i-1), ldx21, work)
			}
			if (*m)-(*q)+1 > i {
				Dlarf('R', toPtr((*m)-(*q)-i+1), toPtr((*m)-(*p)-i+1), x21.Vector(i-1, i-1), ldx21, taup2.GetPtr(i-1), x22.Off(i-1, i-1), ldx22, work)
			}

			if i < (*q) {
				goblas.Dscal(toPtr((*q)-i), toPtrf64(-z1*z3*math.Sin(theta.Get(i-1))), x11.Vector(i+1-1, i-1), func() *int { y := 1; return &y }())
				goblas.Daxpy(toPtr((*q)-i), toPtrf64(z2*z3*math.Cos(theta.Get(i-1))), x21.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), x11.Vector(i+1-1, i-1), func() *int { y := 1; return &y }())
			}
			goblas.Dscal(toPtr((*m)-(*q)-i+1), toPtrf64(-z1*z4*math.Sin(theta.Get(i-1))), x12.Vector(i-1, i-1), func() *int { y := 1; return &y }())
			goblas.Daxpy(toPtr((*m)-(*q)-i+1), toPtrf64(z2*z4*math.Cos(theta.Get(i-1))), x22.Vector(i-1, i-1), func() *int { y := 1; return &y }(), x12.Vector(i-1, i-1), func() *int { y := 1; return &y }())

			if i < (*q) {
				phi.Set(i-1, math.Atan2(goblas.Dnrm2(toPtr((*q)-i), x11.Vector(i+1-1, i-1), func() *int { y := 1; return &y }()), goblas.Dnrm2(toPtr((*m)-(*q)-i+1), x12.Vector(i-1, i-1), func() *int { y := 1; return &y }())))
			}

			if i < (*q) {
				if (*q)-i == 1 {
					Dlarfgp(toPtr((*q)-i), x11.GetPtr(i+1-1, i-1), x11.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), tauq1.GetPtr(i-1))
				} else {
					Dlarfgp(toPtr((*q)-i), x11.GetPtr(i+1-1, i-1), x11.Vector(i+2-1, i-1), func() *int { y := 1; return &y }(), tauq1.GetPtr(i-1))
				}
				x11.Set(i+1-1, i-1, one)
			}
			if (*m)-(*q) > i {
				Dlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr(i-1))
			} else {
				Dlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.Vector(i-1, i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr(i-1))
			}
			x12.Set(i-1, i-1, one)

			if i < (*q) {
				Dlarf('L', toPtr((*q)-i), toPtr((*p)-i), x11.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), tauq1.GetPtr(i-1), x11.Off(i+1-1, i+1-1), ldx11, work)
				Dlarf('L', toPtr((*q)-i), toPtr((*m)-(*p)-i), x11.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), tauq1.GetPtr(i-1), x21.Off(i+1-1, i+1-1), ldx21, work)
			}
			Dlarf('L', toPtr((*m)-(*q)-i+1), toPtr((*p)-i), x12.Vector(i-1, i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr(i-1), x12.Off(i-1, i+1-1), ldx12, work)
			if (*m)-(*p)-i > 0 {
				Dlarf('L', toPtr((*m)-(*q)-i+1), toPtr((*m)-(*p)-i), x12.Vector(i-1, i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr(i-1), x22.Off(i-1, i+1-1), ldx22, work)
			}

		}

		//        Reduce columns Q + 1, ..., P of X12, X22
		for i = (*q) + 1; i <= (*p); i++ {

			goblas.Dscal(toPtr((*m)-(*q)-i+1), toPtrf64(-z1*z4), x12.Vector(i-1, i-1), func() *int { y := 1; return &y }())
			Dlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr(i-1))
			x12.Set(i-1, i-1, one)

			if (*p) > i {
				Dlarf('L', toPtr((*m)-(*q)-i+1), toPtr((*p)-i), x12.Vector(i-1, i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr(i-1), x12.Off(i-1, i+1-1), ldx12, work)
			}
			if (*m)-(*p)-(*q) >= 1 {
				Dlarf('L', toPtr((*m)-(*q)-i+1), toPtr((*m)-(*p)-(*q)), x12.Vector(i-1, i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr(i-1), x22.Off(i-1, (*q)+1-1), ldx22, work)
			}

		}

		//        Reduce columns P + 1, ..., M - Q of X12, X22
		for i = 1; i <= (*m)-(*p)-(*q); i++ {

			goblas.Dscal(toPtr((*m)-(*p)-(*q)-i+1), toPtrf64(z2*z4), x22.Vector((*p)+i-1, (*q)+i-1), func() *int { y := 1; return &y }())
			if (*m)-(*p)-(*q) == i {
				Dlarfgp(toPtr((*m)-(*p)-(*q)-i+1), x22.GetPtr((*p)+i-1, (*q)+i-1), x22.Vector((*p)+i-1, (*q)+i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr((*p)+i-1))
			} else {
				Dlarfgp(toPtr((*m)-(*p)-(*q)-i+1), x22.GetPtr((*p)+i-1, (*q)+i-1), x22.Vector((*p)+i+1-1, (*q)+i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr((*p)+i-1))
				Dlarf('L', toPtr((*m)-(*p)-(*q)-i+1), toPtr((*m)-(*p)-(*q)-i), x22.Vector((*p)+i-1, (*q)+i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr((*p)+i-1), x22.Off((*p)+i-1, (*q)+i+1-1), ldx22, work)
			}
			x22.Set((*p)+i-1, (*q)+i-1, one)

		}

	}
}
