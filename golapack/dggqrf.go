package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dggqrf computes a generalized QR factorization of an N-by-M matrix A
// and an N-by-P matrix B:
//
//             A = Q*R,        B = Q*T*Z,
//
// where Q is an N-by-N orthogonal matrix, Z is a P-by-P orthogonal
// matrix, and R and T assume one of the forms:
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
//              inv(B)*A = Z**T*(inv(T)*R)
//
// where inv(B) denotes the inverse of the matrix B, and Z**T denotes the
// transpose of the matrix Z.
func Dggqrf(n, m, p int, a *mat.Matrix, taua *mat.Vector, b *mat.Matrix, taub, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var lopt, lwkopt, nb, nb1, nb2, nb3 int

	//     Test the input parameters
	nb1 = Ilaenv(1, "Dgeqrf", []byte{' '}, n, m, -1, -1)
	nb2 = Ilaenv(1, "Dgerqf", []byte{' '}, n, p, -1, -1)
	nb3 = Ilaenv(1, "Dormqr", []byte{' '}, n, m, p, -1)
	nb = max(nb1, nb2, nb3)
	lwkopt = max(n, m, p) * nb
	work.Set(0, float64(lwkopt))
	lquery = (lwork == -1)
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if p < 0 {
		err = fmt.Errorf("p < 0: p=%v", p)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if lwork < max(1, n, m, p) && !lquery {
		err = fmt.Errorf("lwork < max(1, n, m, p) && !lquery: lwork=%v, m=%v, n=%v, p=%v, lquery=%v", lwork, m, n, p, lquery)
	}
	if err != nil {
		gltest.Xerbla2("Dggqrf", err)
		return
	} else if lquery {
		return
	}

	//     QR factorization of N-by-M matrix A: A = Q*R
	if err = Dgeqrf(n, m, a, taua, work, lwork); err != nil {
		panic(err)
	}
	lopt = int(work.Get(0))

	//     Update B := Q**T*B.
	if err = Dormqr(Left, Trans, n, p, min(n, m), a, taua, b, work, lwork); err != nil {
		panic(err)
	}
	lopt = max(lopt, int(work.Get(0)))

	//     RQ factorization of N-by-P matrix B: B = T*Z.
	if err = Dgerqf(n, p, b, taub, work, lwork); err != nil {
		panic(err)
	}
	work.Set(0, float64(max(lopt, int(work.Get(0)))))

	return
}
