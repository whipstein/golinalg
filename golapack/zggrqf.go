package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zggrqf computes a generalized RQ factorization of an M-by-N matrix A
// and a P-by-N matrix B:
//
//             A = R*Q,        B = Z*T*Q,
//
// where Q is an N-by-N unitary matrix, Z is a P-by-P unitary
// matrix, and R and T assume one of the forms:
//
// if M <= N,  R = ( 0  R12 ) M,   or if M > N,  R = ( R11 ) M-N,
//                  N-M  M                           ( R21 ) N
//                                                      N
//
// where R12 or R21 is upper triangular, and
//
// if P >= N,  T = ( T11 ) N  ,   or if P < N,  T = ( T11  T12 ) P,
//                 (  0  ) P-N                         P   N-P
//                    N
//
// where T11 is upper triangular.
//
// In particular, if B is square and nonsingular, the GRQ factorization
// of A and B implicitly gives the RQ factorization of A*inv(B):
//
//              A*inv(B) = (R*inv(T))*Z**H
//
// where inv(B) denotes the inverse of the matrix B, and Z**H denotes the
// conjugate transpose of the matrix Z.
func Zggrqf(m, p, n *int, a *mat.CMatrix, lda *int, taua *mat.CVector, b *mat.CMatrix, ldb *int, taub, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var lopt, lwkopt, nb, nb1, nb2, nb3 int

	//     Test the input parameters
	(*info) = 0
	nb1 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGERQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
	nb2 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, p, n, toPtr(-1), toPtr(-1))
	nb3 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMRQ"), []byte{' '}, m, n, p, toPtr(-1))
	nb = max(nb1, nb2, nb3)
	lwkopt = max(*n, *m, *p) * nb
	work.SetRe(0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*p) < 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < max(1, *m) {
		(*info) = -5
	} else if (*ldb) < max(1, *p) {
		(*info) = -8
	} else if (*lwork) < max(1, *m, *p, *n) && !lquery {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGGRQF"), -(*info))
		return
	} else if lquery {
		return
	}

	//     RQ factorization of M-by-N matrix A: A = R*Q
	Zgerqf(m, n, a, lda, taua, work, lwork, info)
	lopt = int(work.GetRe(0))

	//     Update B := B*Q**H
	Zunmrq('R', 'C', p, n, toPtr(min(*m, *n)), a.Off(max(1, (*m)-(*n)+1)-1, 0), lda, taua, b, ldb, work, lwork, info)
	lopt = max(lopt, int(work.GetRe(0)))

	//     QR factorization of P-by-N matrix B: B = Z*T
	Zgeqrf(p, n, b, ldb, taub, work, lwork, info)
	work.SetRe(0, float64(max(lopt, int(work.GetRe(0)))))
}
