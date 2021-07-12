package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dggglm solves a general Gauss-Markov linear model (GLM) problem:
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
func Dggglm(n, m, p *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, d, x, y, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var one, zero float64
	var i, lopt, lwkmin, lwkopt, nb, nb1, nb2, nb3, nb4, np int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

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
			nb1 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEQRF"), []byte{' '}, n, m, toPtr(-1), toPtr(-1))
			nb2 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGERQF"), []byte{' '}, n, m, toPtr(-1), toPtr(-1))
			nb3 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMQR"), []byte{' '}, n, m, p, toPtr(-1))
			nb4 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMRQ"), []byte{' '}, n, m, p, toPtr(-1))
			nb = max(nb1, nb2, nb3, nb4)
			lwkmin = (*m) + (*n) + (*p)
			lwkopt = (*m) + np + max(*n, *p)*nb
		}
		work.Set(0, float64(lwkopt))

		if (*lwork) < lwkmin && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGGGLM"), -(*info))
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
	//          Q**T*A = ( R11 ) M,    Q**T*B*Z**T = ( T11   T12 ) M
	//                   (  0  ) N-M                 (  0    T22 ) N-M
	//                      M                         M+P-N  N-M
	//
	//     where R11 and T22 are upper triangular, and Q and Z are
	//     orthogonal.
	Dggqrf(n, m, p, a, lda, work, b, ldb, work.Off((*m)), work.Off((*m)+np), toPtr((*lwork)-(*m)-np), info)
	lopt = int(work.Get((*m) + np + 1 - 1))

	//     Update left-hand-side vector d = Q**T*d = ( d1 ) M
	//                                               ( d2 ) N-M
	Dormqr('L', 'T', n, func() *int { y := 1; return &y }(), m, a, lda, work, d.Matrix(max(1, *n), opts), toPtr(max(1, *n)), work.Off((*m)+np), toPtr((*lwork)-(*m)-np), info)
	lopt = max(lopt, int(work.Get((*m)+np)))

	//     Solve T22*y2 = d2 for y2
	if (*n) > (*m) {
		Dtrtrs('U', 'N', 'N', toPtr((*n)-(*m)), func() *int { y := 1; return &y }(), b.Off((*m), (*m)+(*p)-(*n)), ldb, d.MatrixOff((*m), (*n)-(*m), opts), toPtr((*n)-(*m)), info)

		if (*info) > 0 {
			(*info) = 1
			return
		}

		goblas.Dcopy((*n)-(*m), d.Off((*m), 1), y.Off((*m)+(*p)-(*n), 1))
	}

	//     Set y1 = 0
	for i = 1; i <= (*m)+(*p)-(*n); i++ {
		y.Set(i-1, zero)
	}

	//     Update d1 = d1 - T12*y2
	err = goblas.Dgemv(NoTrans, *m, (*n)-(*m), -one, b.Off(0, (*m)+(*p)-(*n)), y.Off((*m)+(*p)-(*n), 1), one, d.Off(0, 1))

	//     Solve triangular system: R11*x = d1
	if (*m) > 0 {
		Dtrtrs('U', 'N', 'N', m, func() *int { y := 1; return &y }(), a, lda, d.Matrix(*m, opts), m, info)

		if (*info) > 0 {
			(*info) = 2
			return
		}

		//        Copy D to X
		goblas.Dcopy(*m, d.Off(0, 1), x.Off(0, 1))
	}

	//     Backward transformation y = Z**T *y
	Dormrq('L', 'T', p, func() *int { y := 1; return &y }(), &np, b.Off(max(1, (*n)-(*p)+1)-1, 0), ldb, work.Off((*m)), y.Matrix(max(1, *p), opts), toPtr(max(1, *p)), work.Off((*m)+np), toPtr((*lwork)-(*m)-np), info)
	work.Set(0, float64((*m)+np+max(lopt, int(work.Get((*m)+np)))))
}
