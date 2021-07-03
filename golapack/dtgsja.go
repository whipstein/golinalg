package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtgsja computes the generalized singular value decomposition (GSVD)
// of two real upper triangular (or trapezoidal) matrices A and B.
//
// On entry, it is assumed that matrices A and B have the following
// forms, which may be obtained by the preprocessing subroutine DGGSVP
// from a general M-by-N matrix A and P-by-N matrix B:
//
//              N-K-L  K    L
//    A =    K ( 0    A12  A13 ) if M-K-L >= 0;
//           L ( 0     0   A23 )
//       M-K-L ( 0     0    0  )
//
//            N-K-L  K    L
//    A =  K ( 0    A12  A13 ) if M-K-L < 0;
//       M-K ( 0     0   A23 )
//
//            N-K-L  K    L
//    B =  L ( 0     0   B13 )
//       P-L ( 0     0    0  )
//
// where the K-by-K matrix A12 and L-by-L matrix B13 are nonsingular
// upper triangular; A23 is L-by-L upper triangular if M-K-L >= 0,
// otherwise A23 is (M-K)-by-L upper trapezoidal.
//
// On exit,
//
//        U**T *A*Q = D1*( 0 R ),    V**T *B*Q = D2*( 0 R ),
//
// where U, V and Q are orthogonal matrices.
// R is a nonsingular upper triangular matrix, and D1 and D2 are
// ``diagonal'' matrices, which are of the following structures:
//
// If M-K-L >= 0,
//
//                     K  L
//        D1 =     K ( I  0 )
//                 L ( 0  C )
//             M-K-L ( 0  0 )
//
//                   K  L
//        D2 = L   ( 0  S )
//             P-L ( 0  0 )
//
//                N-K-L  K    L
//   ( 0 R ) = K (  0   R11  R12 ) K
//             L (  0    0   R22 ) L
//
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
//                K M-K K+L-M
//     D1 =   K ( I  0    0   )
//          M-K ( 0  C    0   )
//
//                  K M-K K+L-M
//     D2 =   M-K ( 0  S    0   )
//          K+L-M ( 0  0    I   )
//            P-L ( 0  0    0   )
//
//                N-K-L  K   M-K  K+L-M
// ( 0 R ) =    K ( 0    R11  R12  R13  )
//           M-K ( 0     0   R22  R23  )
//         K+L-M ( 0     0    0   R33  )
//
// where
// C = diag( ALPHA(K+1), ... , ALPHA(M) ),
// S = diag( BETA(K+1),  ... , BETA(M) ),
// C**2 + S**2 = I.
//
// R = ( R11 R12 R13 ) is stored in A(1:M, N-K-L+1:N) and R33 is stored
//     (  0  R22 R23 )
// in B(M-K+1:L,N+M-K-L+1:N) on exit.
//
// The computation of the orthogonal transformation matrices U, V or Q
// is optional.  These matrices may either be formed explicitly, or they
// may be postmultiplied into input matrices U1, V1, or Q1.
func Dtgsja(jobu, jobv, jobq byte, m, p, n, k, l *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, tola, tolb *float64, alpha, beta *mat.Vector, u *mat.Matrix, ldu *int, v *mat.Matrix, ldv *int, q *mat.Matrix, ldq *int, work *mat.Vector, ncycle, info *int) {
	var initq, initu, initv, upper, wantq, wantu, wantv bool
	var a1, a2, a3, b1, b2, b3, csq, csu, csv, error, gamma, one, rwk, snq, snu, snv, ssmin, zero float64
	var i, j, kcycle, maxit int

	maxit = 40
	zero = 0.0
	one = 1.0

	//     Decode and test the input parameters
	initu = jobu == 'I'
	wantu = initu || jobu == 'U'

	initv = jobv == 'I'
	wantv = initv || jobv == 'V'

	initq = jobq == 'I'
	wantq = initq || jobq == 'Q'

	(*info) = 0
	if !(initu || wantu || jobu == 'N') {
		(*info) = -1
	} else if !(initv || wantv || jobv == 'N') {
		(*info) = -2
	} else if !(initq || wantq || jobq == 'N') {
		(*info) = -3
	} else if (*m) < 0 {
		(*info) = -4
	} else if (*p) < 0 {
		(*info) = -5
	} else if (*n) < 0 {
		(*info) = -6
	} else if (*lda) < maxint(1, *m) {
		(*info) = -10
	} else if (*ldb) < maxint(1, *p) {
		(*info) = -12
	} else if (*ldu) < 1 || (wantu && (*ldu) < (*m)) {
		(*info) = -18
	} else if (*ldv) < 1 || (wantv && (*ldv) < (*p)) {
		(*info) = -20
	} else if (*ldq) < 1 || (wantq && (*ldq) < (*n)) {
		(*info) = -22
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTGSJA"), -(*info))
		return
	}

	//     Initialize U, V and Q, if necessary
	if initu {
		Dlaset('F', m, m, &zero, &one, u, ldu)
	}
	if initv {
		Dlaset('F', p, p, &zero, &one, v, ldv)
	}
	if initq {
		Dlaset('F', n, n, &zero, &one, q, ldq)
	}

	//     Loop until convergence
	upper = false
	for kcycle = 1; kcycle <= maxit; kcycle++ {

		upper = !upper

		for i = 1; i <= (*l)-1; i++ {
			for j = i + 1; j <= (*l); j++ {

				a1 = zero
				a2 = zero
				a3 = zero
				if (*k)+i <= (*m) {
					a1 = a.Get((*k)+i-1, (*n)-(*l)+i-1)
				}
				if (*k)+j <= (*m) {
					a3 = a.Get((*k)+j-1, (*n)-(*l)+j-1)
				}

				b1 = b.Get(i-1, (*n)-(*l)+i-1)
				b3 = b.Get(j-1, (*n)-(*l)+j-1)

				if upper {
					if (*k)+i <= (*m) {
						a2 = a.Get((*k)+i-1, (*n)-(*l)+j-1)
					}
					b2 = b.Get(i-1, (*n)-(*l)+j-1)
				} else {
					if (*k)+j <= (*m) {
						a2 = a.Get((*k)+j-1, (*n)-(*l)+i-1)
					}
					b2 = b.Get(j-1, (*n)-(*l)+i-1)
				}

				Dlags2(upper, &a1, &a2, &a3, &b1, &b2, &b3, &csu, &snu, &csv, &snv, &csq, &snq)

				//              Update (K+I)-th and (K+J)-th rows of matrix A: U**T *A
				if (*k)+j <= (*m) {
					goblas.Drot(*l, a.Vector((*k)+j-1, (*n)-(*l)+1-1), *lda, a.Vector((*k)+i-1, (*n)-(*l)+1-1), *lda, csu, snu)
				}

				//              Update I-th and J-th rows of matrix B: V**T *B
				goblas.Drot(*l, b.Vector(j-1, (*n)-(*l)+1-1), *ldb, b.Vector(i-1, (*n)-(*l)+1-1), *ldb, csv, snv)

				//              Update (N-L+I)-th and (N-L+J)-th columns of matrices
				//              A and B: A*Q and B*Q
				goblas.Drot(minint((*k)+(*l), *m), a.Vector(0, (*n)-(*l)+j-1), 1, a.Vector(0, (*n)-(*l)+i-1), 1, csq, snq)

				goblas.Drot(*l, b.Vector(0, (*n)-(*l)+j-1), 1, b.Vector(0, (*n)-(*l)+i-1), 1, csq, snq)

				if upper {
					if (*k)+i <= (*m) {
						a.Set((*k)+i-1, (*n)-(*l)+j-1, zero)
					}
					b.Set(i-1, (*n)-(*l)+j-1, zero)
				} else {
					if (*k)+j <= (*m) {
						a.Set((*k)+j-1, (*n)-(*l)+i-1, zero)
					}
					b.Set(j-1, (*n)-(*l)+i-1, zero)
				}

				//              Update orthogonal matrices U, V, Q, if desired.
				if wantu && (*k)+j <= (*m) {
					goblas.Drot(*m, u.Vector(0, (*k)+j-1), 1, u.Vector(0, (*k)+i-1), 1, csu, snu)
				}

				if wantv {
					goblas.Drot(*p, v.Vector(0, j-1), 1, v.Vector(0, i-1), 1, csv, snv)
				}

				if wantq {
					goblas.Drot(*n, q.Vector(0, (*n)-(*l)+j-1), 1, q.Vector(0, (*n)-(*l)+i-1), 1, csq, snq)
				}

			}
		}

		if !upper {
			//           The matrices A13 and B13 were lower triangular at the start
			//           of the cycle, and are now upper triangular.
			//
			//           Convergence test: test the parallelism of the corresponding
			//           rows of A and B.
			error = zero
			for i = 1; i <= minint(*l, (*m)-(*k)); i++ {
				goblas.Dcopy((*l)-i+1, a.Vector((*k)+i-1, (*n)-(*l)+i-1), *lda, work, 1)
				goblas.Dcopy((*l)-i+1, b.Vector(i-1, (*n)-(*l)+i-1), *ldb, work.Off((*l)+1-1), 1)
				Dlapll(toPtr((*l)-i+1), work, toPtr(1), work.Off((*l)+1-1), toPtr(1), &ssmin)
				error = maxf64(error, ssmin)
			}

			if math.Abs(error) <= minf64(*tola, *tolb) {
				goto label50
			}
		}

		//        End of cycle loop
	}

	//     The algorithm has not converged after MAXIT cycles.
	(*info) = 1
	goto label100

label50:
	;

	//     If ERROR <= MIN(TOLA,TOLB), then the algorithm has converged.
	//     Compute the generalized singular value pairs (ALPHA, BETA), and
	//     set the triangular matrix R to array A.
	for i = 1; i <= (*k); i++ {
		alpha.Set(i-1, one)
		beta.Set(i-1, zero)
	}

	for i = 1; i <= minint(*l, (*m)-(*k)); i++ {

		a1 = a.Get((*k)+i-1, (*n)-(*l)+i-1)
		b1 = b.Get(i-1, (*n)-(*l)+i-1)

		if a1 != zero {
			gamma = b1 / a1

			//           change sign if necessary
			if gamma < zero {
				goblas.Dscal((*l)-i+1, -one, b.Vector(i-1, (*n)-(*l)+i-1), *ldb)
				if wantv {
					goblas.Dscal(*p, -one, v.Vector(0, i-1), 1)
				}
			}

			Dlartg(toPtrf64(math.Abs(gamma)), &one, beta.GetPtr((*k)+i-1), alpha.GetPtr((*k)+i-1), &rwk)

			if alpha.Get((*k)+i-1) >= beta.Get((*k)+i-1) {
				goblas.Dscal((*l)-i+1, one/alpha.Get((*k)+i-1), a.Vector((*k)+i-1, (*n)-(*l)+i-1), *lda)
			} else {
				goblas.Dscal((*l)-i+1, one/beta.Get((*k)+i-1), b.Vector(i-1, (*n)-(*l)+i-1), *ldb)
				goblas.Dcopy((*l)-i+1, b.Vector(i-1, (*n)-(*l)+i-1), *ldb, a.Vector((*k)+i-1, (*n)-(*l)+i-1), *lda)
			}

		} else {

			alpha.Set((*k)+i-1, zero)
			beta.Set((*k)+i-1, one)
			goblas.Dcopy((*l)-i+1, b.Vector(i-1, (*n)-(*l)+i-1), *ldb, a.Vector((*k)+i-1, (*n)-(*l)+i-1), *lda)

		}

	}

	//     Post-assignment
	for i = (*m) + 1; i <= (*k)+(*l); i++ {
		alpha.Set(i-1, zero)
		beta.Set(i-1, one)
	}

	if (*k)+(*l) < (*n) {
		for i = (*k) + (*l) + 1; i <= (*n); i++ {
			alpha.Set(i-1, zero)
			beta.Set(i-1, zero)
		}
	}

label100:
	;
	(*ncycle) = kcycle
}
