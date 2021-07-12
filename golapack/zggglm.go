package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zggglm solves a general Gauss-Markov linear model (GLM) problem:
//
//         minimize || y ||_2   subject to   d = A*x + B*y
//             x
//
// where A is an N-by-M matrix, B is an N-by-P matrix, and d is a
// given N-vector. It is assumed that M <= N <= M+P, and
//
//            rank(A) = M    and    rank( A B ) = N.
//
// Under these assumptions, the constrained equation is always
// consistent, and there is a unique solution x and a minimal 2-norm
// solution y, which is obtained using a generalized QR factorization
// of the matrices (A, B) given by
//
//    A = Q*(R),   B = Q*T*Z.
//          (0)
//
// In particular, if matrix B is square nonsingular, then the problem
// GLM is equivalent to the following weighted linear least squares
// problem
//
//              minimize || inv(B)*(d-A*x) ||_2
//                  x
//
// where inv(B) denotes the inverse of B.
func Zggglm(n, m, p *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, d, x, y, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var cone, czero complex128
	var i, lopt, lwkmin, lwkopt, nb, nb1, nb2, nb3, nb4, np int
	var err error
	_ = err

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	(*info) = 0
	np = min(*n, *p)
	lquery = ((*lwork) == -1)
	if (*n) < 0 {
		(*info) = -1
	} else if (*m) < 0 || (*m) > (*n) {
		(*info) = -2
	} else if (*p) < 0 || (*p) < (*n)-(*m) {
		(*info) = -3
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	} else if (*ldb) < max(1, *n) {
		(*info) = -7
	}

	//     Calculate workspace
	if (*info) == 0 {
		if (*n) == 0 {
			lwkmin = 1
			lwkopt = 1
		} else {
			nb1 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, n, m, toPtr(-1), toPtr(-1))
			nb2 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGERQF"), []byte{' '}, n, m, toPtr(-1), toPtr(-1))
			nb3 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte{' '}, n, m, p, toPtr(-1))
			nb4 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMRQ"), []byte{' '}, n, m, p, toPtr(-1))
			nb = max(nb1, nb2, nb3, nb4)
			lwkmin = (*m) + (*n) + (*p)
			lwkopt = (*m) + np + max(*n, *p)*nb
		}
		work.SetRe(0, float64(lwkopt))

		if (*lwork) < lwkmin && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGGGLM"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Compute the GQR factorization of matrices A and B:
	//
	//          Q**H*A = ( R11 ) M,    Q**H*B*Z**H = ( T11   T12 ) M
	//                   (  0  ) N-M                 (  0    T22 ) N-M
	//                      M                         M+P-N  N-M
	//
	//     where R11 and T22 are upper triangular, and Q and Z are
	//     unitary.
	Zggqrf(n, m, p, a, lda, work, b, ldb, work.Off((*m)), work.Off((*m)+np), toPtr((*lwork)-(*m)-np), info)
	lopt = int(work.GetRe((*m) + np + 1 - 1))

	//     Update left-hand-side vector d = Q**H*d = ( d1 ) M
	//                                               ( d2 ) N-M
	Zunmqr('L', 'C', n, func() *int { y := 1; return &y }(), m, a, lda, work, d.CMatrix(max(1, *n), opts), toPtr(max(1, *n)), work.Off((*m)+np), toPtr((*lwork)-(*m)-np), info)
	lopt = max(lopt, int(work.GetRe((*m)+np)))

	//     Solve T22*y2 = d2 for y2
	if (*n) > (*m) {
		Ztrtrs('U', 'N', 'N', toPtr((*n)-(*m)), func() *int { y := 1; return &y }(), b.Off((*m), (*m)+(*p)-(*n)), ldb, d.CMatrixOff((*m), (*n)-(*m), opts), toPtr((*n)-(*m)), info)

		if (*info) > 0 {
			(*info) = 1
			return
		}

		goblas.Zcopy((*n)-(*m), d.Off((*m), 1), y.Off((*m)+(*p)-(*n), 1))
	}

	//     Set y1 = 0
	for i = 1; i <= (*m)+(*p)-(*n); i++ {
		y.Set(i-1, czero)
	}

	//     Update d1 = d1 - T12*y2
	err = goblas.Zgemv(NoTrans, *m, (*n)-(*m), -cone, b.Off(0, (*m)+(*p)-(*n)), y.Off((*m)+(*p)-(*n), 1), cone, d.Off(0, 1))

	//     Solve triangular system: R11*x = d1
	if (*m) > 0 {
		Ztrtrs('U', 'N', 'N', m, func() *int { y := 1; return &y }(), a, lda, d.CMatrix(*m, opts), m, info)

		if (*info) > 0 {
			(*info) = 2
			return
		}

		//        Copy D to X
		goblas.Zcopy(*m, d.Off(0, 1), x.Off(0, 1))
	}

	//     Backward transformation y = Z**H *y
	Zunmrq('L', 'C', p, func() *int { y := 1; return &y }(), &np, b.Off(max(1, (*n)-(*p)+1)-1, 0), ldb, work.Off((*m)), y.CMatrix(max(1, *p), opts), toPtr(max(1, *p)), work.Off((*m)+np), toPtr((*lwork)-(*m)-np), info)
	work.SetRe(0, float64((*m)+np+max(lopt, int(work.GetRe((*m)+np)))))
}
