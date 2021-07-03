package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgglse solves the linear equality-constrained least squares (LSE)
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
func Zgglse(m, n, p *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, c, d, x, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var cone complex128
	var lopt, lwkmin, lwkopt, mn, nb, nb1, nb2, nb3, nb4, nr int
	var err error
	_ = err

	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	(*info) = 0
	mn = minint(*m, *n)
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*p) < 0 || (*p) > (*n) || (*p) < (*n)-(*m) {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *p) {
		(*info) = -7
	}

	//     Calculate workspace
	if (*info) == 0 {
		if (*n) == 0 {
			lwkmin = 1
			lwkopt = 1
		} else {
			nb1 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
			nb2 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGERQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
			nb3 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte{' '}, m, n, p, toPtr(-1))
			nb4 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMRQ"), []byte{' '}, m, n, p, toPtr(-1))
			nb = maxint(nb1, nb2, nb3, nb4)
			lwkmin = (*m) + (*n) + (*p)
			lwkopt = (*p) + mn + maxint(*m, *n)*nb
		}
		work.SetRe(0, float64(lwkopt))

		if (*lwork) < lwkmin && !lquery {
			(*info) = -12
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGGLSE"), -(*info))
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
	//            B*Q**H = (  0  T12 ) P   Z**H*A*Q**H = ( R11 R12 ) N-P
	//                        N-P  P                     (  0  R22 ) M+P-N
	//                                                      N-P  P
	//
	//     where T12 and R11 are upper triangular, and Q and Z are
	//     unitary.
	Zggrqf(p, m, n, b, ldb, work, a, lda, work.Off((*p)+1-1), work.Off((*p)+mn+1-1), toPtr((*lwork)-(*p)-mn), info)
	lopt = int(work.GetRe((*p) + mn + 1 - 1))

	//     Update c = Z**H *c = ( c1 ) N-P
	//                       ( c2 ) M+P-N
	Zunmqr('L', 'C', m, func() *int { y := 1; return &y }(), &mn, a, lda, work.Off((*p)+1-1), c.CMatrix(maxint(1, *m), opts), toPtr(maxint(1, *m)), work.Off((*p)+mn+1-1), toPtr((*lwork)-(*p)-mn), info)
	lopt = maxint(lopt, int(work.GetRe((*p)+mn+1-1)))

	//     Solve T12*x2 = d for x2
	if (*p) > 0 {
		Ztrtrs('U', 'N', 'N', p, func() *int { y := 1; return &y }(), b.Off(0, (*n)-(*p)+1-1), ldb, d.CMatrix(*p, opts), p, info)

		if (*info) > 0 {
			(*info) = 1
			return
		}

		//        Put the solution in X
		goblas.Zcopy(*p, d, 1, x.Off((*n)-(*p)+1-1), 1)

		//        Update c1
		err = goblas.Zgemv(NoTrans, (*n)-(*p), *p, -cone, a.Off(0, (*n)-(*p)+1-1), *lda, d, 1, cone, c, 1)
	}
	//
	//     Solve R11*x1 = c1 for x1
	//
	if (*n) > (*p) {
		Ztrtrs('U', 'N', 'N', toPtr((*n)-(*p)), func() *int { y := 1; return &y }(), a, lda, c.CMatrix((*n)-(*p), opts), toPtr((*n)-(*p)), info)

		if (*info) > 0 {
			(*info) = 2
			return
		}

		//        Put the solutions in X
		goblas.Zcopy((*n)-(*p), c, 1, x, 1)
	}

	//     Compute the residual vector:
	if (*m) < (*n) {
		nr = (*m) + (*p) - (*n)
		if nr > 0 {
			err = goblas.Zgemv(NoTrans, nr, (*n)-(*m), -cone, a.Off((*n)-(*p)+1-1, (*m)+1-1), *lda, d.Off(nr+1-1), 1, cone, c.Off((*n)-(*p)+1-1), 1)
		}
	} else {
		nr = (*p)
	}
	if nr > 0 {
		err = goblas.Ztrmv(Upper, NoTrans, NonUnit, nr, a.Off((*n)-(*p)+1-1, (*n)-(*p)+1-1), *lda, d, 1)
		goblas.Zaxpy(nr, -cone, d, 1, c.Off((*n)-(*p)+1-1), 1)
	}
	//
	//     Backward transformation x = Q**H*x
	//
	Zunmrq('L', 'C', n, func() *int { y := 1; return &y }(), p, b, ldb, work.Off(0), x.CMatrix(*n, opts), n, work.Off((*p)+mn+1-1), toPtr((*lwork)-(*p)-mn), info)
	work.SetRe(0, float64((*p)+mn+maxint(lopt, int(work.GetRe((*p)+mn+1-1)))))
}
