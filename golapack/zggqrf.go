package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zggqrf computes a generalized QR factorization of an N-by-M matrix A
// and an N-by-P matrix B:
//
//             A = Q*R,        B = Q*T*Z,
//
// where Q is an N-by-N unitary matrix, Z is a P-by-P unitary matrix,
// and R and T assume one of the forms:
//
// if N >= M,  R = ( R11 ) M  ,   or if N < M,  R = ( R11  R12 ) N,
//                 (  0  ) N-M                         N   M-N
//                    M
//
// where R11 is upper triangular, and
//
// if N <= P,  T = ( 0  T12 ) N,   or if N > P,  T = ( T11 ) N-P,
//                  P-N  N                           ( T21 ) P
//                                                      P
//
// where T12 or T21 is upper triangular.
//
// In particular, if B is square and nonsingular, the GQR factorization
// of A and B implicitly gives the QR factorization of inv(B)*A:
//
//              inv(B)*A = Z**H * (inv(T)*R)
//
// where inv(B) denotes the inverse of the matrix B, and Z**H denotes the
// conjugate transpose of matrix Z.
func Zggqrf(n, m, p *int, a *mat.CMatrix, lda *int, taua *mat.CVector, b *mat.CMatrix, ldb *int, taub, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var lopt, lwkopt, nb, nb1, nb2, nb3 int

	//     Test the input parameters
	(*info) = 0
	nb1 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, n, m, toPtr(-1), toPtr(-1))
	nb2 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGERQF"), []byte{' '}, n, p, toPtr(-1), toPtr(-1))
	nb3 = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNMQR"), []byte{' '}, n, m, p, toPtr(-1))
	nb = maxint(nb1, nb2, nb3)
	lwkopt = maxint(*n, *m, *p) * nb
	work.SetRe(0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*n) < 0 {
		(*info) = -1
	} else if (*m) < 0 {
		(*info) = -2
	} else if (*p) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	} else if (*lwork) < maxint(1, *n, *m, *p) && !lquery {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGGQRF"), -(*info))
		return
	} else if lquery {
		return
	}

	//     QR factorization of N-by-M matrix A: A = Q*R
	Zgeqrf(n, m, a, lda, taua, work, lwork, info)
	lopt = int(work.GetRe(0))

	//     Update B := Q**H*B.
	Zunmqr('L', 'C', n, p, toPtr(minint(*n, *m)), a, lda, taua, b, ldb, work, lwork, info)
	lopt = maxint(lopt, int(work.GetRe(0)))

	//     RQ factorization of N-by-P matrix B: B = T*Z.
	Zgerqf(n, p, b, ldb, taub, work, lwork, info)
	work.SetRe(0, float64(maxint(lopt, int(work.GetRe(0)))))
}
