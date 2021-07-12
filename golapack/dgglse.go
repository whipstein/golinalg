package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgglse solves the linear equality-constrained least squares (LSE)
// problem:
//
//         minimize || c - A*x ||_2   subject to   B*x = d
//
// where A is an M-by-N matrix, B is a P-by-N matrix, c is a given
// M-vector, and d is a given P-vector. It is assumed that
// P <= N <= M+P, and
//
//          rank(B) = P and  rank( (A) ) = N.
//                               ( (B) )
//
// These conditions ensure that the LSE problem has a unique solution,
// which is obtained using a generalized RQ factorization of the
// matrices (B, A) given by
//
//    B = (0 R)*Q,   A = Z*T*Q.
func Dgglse(m, n, p *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, c, d, x, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var one float64
	var lopt, lwkmin, lwkopt, mn, nb, nb1, nb2, nb3, nb4, nr int
	var err error
	_ = err

	one = 1.0

	//     Test the input parameters
	(*info) = 0
	mn = min(*m, *n)
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*p) < 0 || (*p) > (*n) || (*p) < (*n)-(*m) {
		(*info) = -3
	} else if (*lda) < max(1, *m) {
		(*info) = -5
	} else if (*ldb) < max(1, *p) {
		(*info) = -7
	}

	//     Calculate workspace
	if (*info) == 0 {
		if (*n) == 0 {
			lwkmin = 1
			lwkopt = 1
		} else {
			nb1 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
			nb2 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGERQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
			nb3 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMQR"), []byte{' '}, m, n, p, toPtr(-1))
			nb4 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMRQ"), []byte{' '}, m, n, p, toPtr(-1))
			nb = max(nb1, nb2, nb3, nb4)
			lwkmin = (*m) + (*n) + (*p)
			lwkopt = (*p) + mn + max(*m, *n)*nb
		}
		work.Set(0, float64(lwkopt))

		if (*lwork) < lwkmin && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DGGLSE"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Compute the GRQ factorization of matrices B and A:
	//
	//            B*Q**T = (  0  T12 ) P   Z**T*A*Q**T = ( R11 R12 ) N-P
	//                        N-P  P                     (  0  R22 ) M+P-N
	//                                                      N-P  P
	//
	//     where T12 and R11 are upper triangular, and Q and Z are
	//     orthogonal.
	Dggrqf(p, m, n, b, ldb, work, a, lda, work.Off((*p)), work.Off((*p)+mn), toPtr((*lwork)-(*p)-mn), info)
	lopt = int(work.Get((*p) + mn + 1 - 1))

	//     Update c = Z**T *c = ( c1 ) N-P
	//                          ( c2 ) M+P-N
	Dormqr('L', 'T', m, func() *int { y := 1; return &y }(), &mn, a, lda, work.Off((*p)), c.Matrix(max(1, *m), opts), toPtr(max(1, *m)), work.Off((*p)+mn), toPtr((*lwork)-(*p)-mn), info)
	lopt = max(lopt, int(work.Get((*p)+mn)))

	//     Solve T12*x2 = d for x2
	if (*p) > 0 {
		Dtrtrs('U', 'N', 'N', p, func() *int { y := 1; return &y }(), b.Off(0, (*n)-(*p)), ldb, d.Matrix(*p, opts), p, info)

		if (*info) > 0 {
			(*info) = 1
			return
		}

		//        Put the solution in X
		goblas.Dcopy(*p, d.Off(0, 1), x.Off((*n)-(*p), 1))

		//        Update c1
		err = goblas.Dgemv(NoTrans, (*n)-(*p), *p, -one, a.Off(0, (*n)-(*p)), d.Off(0, 1), one, c.Off(0, 1))
	}

	//     Solve R11*x1 = c1 for x1
	if (*n) > (*p) {
		Dtrtrs('U', 'N', 'N', toPtr((*n)-(*p)), func() *int { y := 1; return &y }(), a, lda, c.Matrix((*n)-(*p), opts), toPtr((*n)-(*p)), info)

		if (*info) > 0 {
			(*info) = 2
			return
		}

		//        Put the solutions in X
		goblas.Dcopy((*n)-(*p), c.Off(0, 1), x.Off(0, 1))
	}

	//     Compute the residual vector:
	if (*m) < (*n) {
		nr = (*m) + (*p) - (*n)
		if nr > 0 {
			err = goblas.Dgemv(NoTrans, nr, (*n)-(*m), -one, a.Off((*n)-(*p), (*m)), d.Off(nr, 1), one, c.Off((*n)-(*p), 1))
		}
	} else {
		nr = (*p)
	}
	if nr > 0 {
		err = goblas.Dtrmv(Upper, NoTrans, NonUnit, nr, a.Off((*n)-(*p), (*n)-(*p)), d.Off(0, 1))
		goblas.Daxpy(nr, -one, d.Off(0, 1), c.Off((*n)-(*p), 1))
	}

	//     Backward transformation x = Q**T*x
	Dormrq('L', 'T', n, func() *int { y := 1; return &y }(), p, b, ldb, work, x.Matrix(*n, opts), n, work.Off((*p)+mn), toPtr((*lwork)-(*p)-mn), info)
	work.Set(0, float64((*p)+mn+max(lopt, int(work.Get((*p)+mn)))))
}
