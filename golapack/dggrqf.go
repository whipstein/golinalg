package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dggrqf computes a generalized RQ factorization of an M-by-N matrix A
// and a P-by-N matrix B:
//
//             A = R*Q,        B = Z*T*Q,
//
// where Q is an N-by-N orthogonal matrix, Z is a P-by-P orthogonal
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
//              A*inv(B) = (R*inv(T))*Z**T
//
// where inv(B) denotes the inverse of the matrix B, and Z**T denotes the
// transpose of the matrix Z.
func Dggrqf(m, p, n *int, a *mat.Matrix, lda *int, taua *mat.Vector, b *mat.Matrix, ldb *int, taub *mat.Vector, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var lopt, lwkopt, nb, nb1, nb2, nb3 int

	//     Test the input parameters
	(*info) = 0
	nb1 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGERQF"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
	nb2 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEQRF"), []byte{' '}, p, n, toPtr(-1), toPtr(-1))
	nb3 = Ilaenv(func() *int { y := 1; return &y }(), []byte("DORMRQ"), []byte{' '}, m, n, p, toPtr(-1))
	nb = maxint(nb1, nb2, nb3)
	lwkopt = maxint(*n, *m, *p) * nb
	work.Set(0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*p) < 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *p) {
		(*info) = -8
	} else if (*lwork) < maxint(1, *m, *p, *n) && !lquery {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGGRQF"), -(*info))
		return
	} else if lquery {
		return
	}

	//     RQ factorization of M-by-N matrix A: A = R*Q
	Dgerqf(m, n, a, lda, taua, work, lwork, info)
	lopt = int(work.Get(0))

	//     Update B := B*Q**T
	Dormrq('R', 'T', p, n, toPtr(minint(*m, *n)), a.Off(maxint(1, (*m)-(*n)+1)-1, 0), lda, taua, b, ldb, work, lwork, info)
	lopt = maxint(lopt, int(work.Get(0)))

	//     QR factorization of P-by-N matrix B: B = Z*T
	Dgeqrf(p, n, b, ldb, taub, work, lwork, info)
	work.Set(0, float64(maxint(lopt, int(work.Get(0)))))
}
