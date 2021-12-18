package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zggsvd3 computes the generalized singular value decomposition (GSVD)
// of an M-by-N complex matrix A and P-by-N complex matrix B:
//
//       U**H*A*Q = D1*( 0 R ),    V**H*B*Q = D2*( 0 R )
//
// where U, V and Q are unitary matrices.
// Let K+L = the effective numerical rank of the
// matrix (A**H,B**H)**H, then R is a (K+L)-by-(K+L) nonsingular upper
// triangular matrix, D1 and D2 are M-by-(K+L) and P-by-(K+L) "diagonal"
// matrices and of the following structures, respectively:
//
// If M-K-L >= 0,
//
//                     K  L
//        D1 =     K ( I  0 )
//                 L ( 0  C )
//             M-K-L ( 0  0 )
//
//                   K  L
//        D2 =   L ( 0  S )
//             P-L ( 0  0 )
//
//                 N-K-L  K    L
//   ( 0 R ) = K (  0   R11  R12 )
//             L (  0    0   R22 )
// where
//
//   C = diag( ALPHA(K+1), ... , ALPHA(K+L) ),
//   S = diag( BETA(K+1),  ... , BETA(K+L) ),
//   C**2 + S**2 = I.
//
//   R is stored in A(1:K+L,N-K-L+1:N) on exit.
//
// If M-K-L < 0,
//
//                   K M-K K+L-M
//        D1 =   K ( I  0    0   )
//             M-K ( 0  C    0   )
//
//                     K M-K K+L-M
//        D2 =   M-K ( 0  S    0  )
//             K+L-M ( 0  0    I  )
//               P-L ( 0  0    0  )
//
//                    N-K-L  K   M-K  K+L-M
//   ( 0 R ) =     K ( 0    R11  R12  R13  )
//               M-K ( 0     0   R22  R23  )
//             K+L-M ( 0     0    0   R33  )
//
// where
//
//   C = diag( ALPHA(K+1), ... , ALPHA(M) ),
//   S = diag( BETA(K+1),  ... , BETA(M) ),
//   C**2 + S**2 = I.
//
//   (R11 R12 R13 ) is stored in A(1:M, N-K-L+1:N), and R33 is stored
//   ( 0  R22 R23 )
//   in B(M-K+1:L,N+M-K-L+1:N) on exit.
//
// The routine computes C, S, R, and optionally the unitary
// transformation matrices U, V and Q.
//
// In particular, if B is an N-by-N nonsingular matrix, then the GSVD of
// A and B implicitly gives the SVD of A*inv(B):
//                      A*inv(B) = U*(D1*inv(D2))*V**H.
// If ( A**H,B**H)**H has orthonormal columns, then the GSVD of A and B is also
// equal to the CS decomposition of A and B. Furthermore, the GSVD can
// be used to derive the solution of the eigenvalue problem:
//                      A**H*A x = lambda* B**H*B x.
// In some literature, the GSVD of A and B is presented in the form
//                  U**H*A*X = ( 0 D1 ),   V**H*B*X = ( 0 D2 )
// where U and V are orthogonal and X is nonsingular, and D1 and D2 are
// ``diagonal''.  The former GSVD form can be converted to the latter
// form by taking the nonsingular matrix X as
//
//                       X = Q*(  I   0    )
//                             (  0 inv(R) )
func Zggsvd3(jobu, jobv, jobq byte, m, n, p int, a, b *mat.CMatrix, alpha, beta *mat.Vector, u, v, q *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, iwork *[]int) (k, l, info int, err error) {
	var lquery, wantq, wantu, wantv bool
	var anorm, bnorm, smax, temp, tola, tolb, ulp, unfl float64
	var i, ibnd, isub, j, lwkopt int

	//     Decode and test the input parameters
	wantu = jobu == 'U'
	wantv = jobv == 'V'
	wantq = jobq == 'Q'
	lquery = (lwork == -1)
	lwkopt = 1

	//     Test the input arguments
	if !(wantu || jobu == 'N') {
		err = fmt.Errorf("!(wantu || jobu == 'N'): jobu='%c'", jobu)
	} else if !(wantv || jobv == 'N') {
		err = fmt.Errorf("!(wantv || jobv == 'N'): jobv='%c'", jobv)
	} else if !(wantq || jobq == 'N') {
		err = fmt.Errorf("!(wantq || jobq == 'N'): jobq='%c'", jobq)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if p < 0 {
		err = fmt.Errorf("p < 0: p=%v", p)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, p) {
		err = fmt.Errorf("b.Rows < max(1, p): b.Rows=%v, p=%v", b.Rows, p)
	} else if u.Rows < 1 || (wantu && u.Rows < m) {
		err = fmt.Errorf("u.Rows < 1 || (wantu && u.Rows < m): jobu='%c', u.Rows=%v, m=%v", jobu, u.Rows, m)
	} else if v.Rows < 1 || (wantv && v.Rows < p) {
		err = fmt.Errorf("v.Rows < 1 || (wantv && v.Rows < p): jobv='%c', v.Rows=%v, p=%v", jobv, v.Rows, p)
	} else if q.Rows < 1 || (wantq && q.Rows < n) {
		err = fmt.Errorf("q.Rows < 1 || (wantq && q.Rows < n): jobq='%c', q.Rows=%v, n=%v", jobq, q.Rows, n)
	} else if lwork < 1 && !lquery {
		err = fmt.Errorf("lwork < 1 && !lquery: lwork=%v, lquery=%v", lwork, lquery)
	}

	//     Compute workspace
	if err == nil {
		if k, l, err = Zggsvp3(jobu, jobv, jobq, m, p, n, a, b, tola, tolb, u, v, q, iwork, rwork, work, work, -1); err != nil {
			panic(err)
		}
		lwkopt = n + int(work.GetRe(0))
		lwkopt = max(2*n, lwkopt)
		lwkopt = max(1, lwkopt)
		work.SetRe(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Zggsvd3", err)
		return
	}
	if lquery {
		return
	}

	//     Compute the Frobenius norm of matrices A and B
	anorm = Zlange('1', m, n, a, rwork)
	bnorm = Zlange('1', p, n, b, rwork)

	//     Get machine precision and set up threshold for determining
	//     the effective numerical rank of the matrices A and B.
	ulp = Dlamch(Precision)
	unfl = Dlamch(SafeMinimum)
	tola = float64(max(m, n)) * math.Max(anorm, unfl) * ulp
	tolb = float64(max(p, n)) * math.Max(bnorm, unfl) * ulp

	if k, l, err = Zggsvp3(jobu, jobv, jobq, m, p, n, a, b, tola, tolb, u, v, q, iwork, rwork, work, work.Off(n), lwork-n); err != nil {
		panic(err)
	}

	//     Compute the GSVD of two upper "triangular" matrices
	if _, info, err = Ztgsja(jobu, jobv, jobq, m, p, n, k, l, a, b, tola, tolb, alpha, beta, u, v, q, work); err != nil {
		panic(err)
	}

	//     Sort the singular values and store the pivot indices in IWORK
	//     Copy ALPHA to RWORK, then sort ALPHA in RWORK
	rwork.Copy(n, alpha, 1, 1)
	ibnd = min(l, m-k)
	for i = 1; i <= ibnd; i++ {
		//        Scan for largest ALPHA(K+I)
		isub = i
		smax = rwork.Get(k + i - 1)
		for j = i + 1; j <= ibnd; j++ {
			temp = rwork.Get(k + j - 1)
			if temp > smax {
				isub = j
				smax = temp
			}
		}
		if isub != i {
			rwork.Set(k+isub-1, rwork.Get(k+i-1))
			rwork.Set(k+i-1, smax)
			(*iwork)[k+i-1] = k + isub
		} else {
			(*iwork)[k+i-1] = k + i
		}
	}

	work.SetRe(0, float64(lwkopt))

	return
}
