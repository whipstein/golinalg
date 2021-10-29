package golapack

import (
	"fmt"

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
func Zggrqf(m, p, n int, a *mat.CMatrix, taua *mat.CVector, b *mat.CMatrix, taub, work *mat.CVector, lwork int) (err error) {
	var lquery bool
	var lopt, lwkopt, nb, nb1, nb2, nb3 int

	//     Test the input parameters
	nb1 = Ilaenv(1, "Zgerqf", []byte{' '}, m, n, -1, -1)
	nb2 = Ilaenv(1, "Zgeqrf", []byte{' '}, p, n, -1, -1)
	nb3 = Ilaenv(1, "Zunmrq", []byte{' '}, m, n, p, -1)
	nb = max(nb1, nb2, nb3)
	lwkopt = max(n, m, p) * nb
	work.SetRe(0, float64(lwkopt))
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if p < 0 {
		err = fmt.Errorf("p < 0: p=%v", p)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, p) {
		err = fmt.Errorf("b.Rows < max(1, p): b.Rows=%v, p=%v", b.Rows, p)
	} else if lwork < max(1, m, p, n) && !lquery {
		err = fmt.Errorf("lwork < max(1, m, p, n) && !lquery: lwork=%v, m=%v, p=%v, n=%v, lquery=%v", lwork, m, p, n, lquery)
	}
	if err != nil {
		gltest.Xerbla2("Zggrqf", err)
		return
	} else if lquery {
		return
	}

	//     RQ factorization of M-by-N matrix A: A = R*Q
	if err = Zgerqf(m, n, a, taua, work, lwork); err != nil {
		panic(err)
	}
	lopt = int(work.GetRe(0))

	//     Update B := B*Q**H
	if err = Zunmrq(Right, ConjTrans, p, n, min(m, n), a.Off(max(1, m-n+1)-1, 0), taua, b, work, lwork); err != nil {
		panic(err)
	}
	lopt = max(lopt, int(work.GetRe(0)))

	//     QR factorization of P-by-N matrix B: B = Z*T
	if err = Zgeqrf(p, n, b, taub, work, lwork); err != nil {
		panic(err)
	}
	work.SetRe(0, float64(max(lopt, int(work.GetRe(0)))))

	return
}
