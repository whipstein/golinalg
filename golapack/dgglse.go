package golapack

import (
	"fmt"

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
func Dgglse(m, n, p int, a, b *mat.Matrix, c, d, x, work *mat.Vector, lwork int) (info int, err error) {
	var lquery bool
	var one float64
	var lopt, lwkmin, lwkopt, mn, nb, nb1, nb2, nb3, nb4, nr int

	one = 1.0

	//     Test the input parameters
	mn = min(m, n)
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if p < 0 || p > n || p < n-m {
		err = fmt.Errorf("p < 0 || p > n || p < n-m: m=%v, n=%v, p=%v", m, n, p)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, p) {
		err = fmt.Errorf("b.Rows < max(1, p): b.Rows=%v, p=%v", b.Rows, p)
	}

	//     Calculate workspace
	if err == nil {
		if n == 0 {
			lwkmin = 1
			lwkopt = 1
		} else {
			nb1 = Ilaenv(1, "Dgeqrf", []byte{' '}, m, n, -1, -1)
			nb2 = Ilaenv(1, "Dgerqf", []byte{' '}, m, n, -1, -1)
			nb3 = Ilaenv(1, "Dormqr", []byte{' '}, m, n, p, -1)
			nb4 = Ilaenv(1, "Dormrq", []byte{' '}, m, n, p, -1)
			nb = max(nb1, nb2, nb3, nb4)
			lwkmin = m + n + p
			lwkopt = p + mn + max(m, n)*nb
		}
		work.Set(0, float64(lwkopt))

		if lwork < lwkmin && !lquery {
			err = fmt.Errorf("lwork < lwkmin && !lquery: lwork=%v, lwkmin=%v, lquery=%v", lwork, lwkmin, lquery)
		}
	}

	if err != nil || info != 0 {
		gltest.Xerbla2("Dgglse", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
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
	if err = Dggrqf(p, m, n, b, work, a, work.Off(p), work.Off(p+mn), lwork-p-mn); err != nil {
		panic(err)
	}
	lopt = int(work.Get(p + mn + 1 - 1))

	//     Update c = Z**T *c = ( c1 ) N-P
	//                          ( c2 ) M+P-N
	if err = Dormqr(Left, Trans, m, 1, mn, a, work.Off(p), c.Matrix(max(1, m), opts), work.Off(p+mn), lwork-p-mn); err != nil {
		panic(err)
	}
	lopt = max(lopt, int(work.Get(p+mn)))

	//     Solve T12*x2 = d for x2
	if p > 0 {
		if info, err = Dtrtrs(Upper, NoTrans, NonUnit, p, 1, b.Off(0, n-p), d.Matrix(p, opts)); err != nil {
			panic(err)
		}

		if info > 0 {
			info = 1
			return
		}

		//        Put the solution in X
		x.Off(n-p).Copy(p, d, 1, 1)

		//        Update c1
		err = c.Gemv(NoTrans, n-p, p, -one, a.Off(0, n-p), d, 1, one, 1)
	}

	//     Solve R11*x1 = c1 for x1
	if n > p {
		if info, err = Dtrtrs(Upper, NoTrans, NonUnit, n-p, 1, a, c.Matrix(n-p, opts)); err != nil {
			panic(err)
		}

		if info > 0 {
			info = 2
			return
		}

		//        Put the solutions in X
		x.Copy(n-p, c, 1, 1)
	}

	//     Compute the residual vector:
	if m < n {
		nr = m + p - n
		if nr > 0 {
			err = c.Off(n-p).Gemv(NoTrans, nr, n-m, -one, a.Off(n-p, m), d.Off(nr), 1, one, 1)
		}
	} else {
		nr = p
	}
	if nr > 0 {
		err = d.Trmv(Upper, NoTrans, NonUnit, nr, a.Off(n-p, n-p), 1)
		c.Off(n-p).Axpy(nr, -one, d, 1, 1)
	}

	//     Backward transformation x = Q**T*x
	if err = Dormrq(Left, Trans, n, 1, p, b, work, x.Matrix(n, opts), work.Off(p+mn), lwork-p-mn); err != nil {
		panic(err)
	}
	work.Set(0, float64(p+mn+max(lopt, int(work.Get(p+mn)))))

	return
}
