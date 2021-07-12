package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrfaa computes the factorization of a complex hermitian matrix A
// using the Aasen's algorithm.  The form of the factorization is
//
//    A = U**H*T*U  or  A = L*T*L**H
//
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and T is a hermitian tridiagonal matrix.
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
func Zhetrfaa(uplo byte, n *int, a *mat.CMatrix, lda *int, ipiv *[]int, work *mat.CVector, lwork, info *int) {
	var lquery, upper bool
	var alpha, one complex128
	var j, j1, j2, j3, jb, k1, k2, lwkopt, mj, nb, nj int
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

	//     Determine the block size
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZHETRF_AA"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	lquery = ((*lwork) == -1)
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *n) {
		(*info) = -4
	} else if (*lwork) < max(1, 2*(*n)) && !lquery {
		(*info) = -7
	}
	//
	if (*info) == 0 {
		lwkopt = (nb + 1) * (*n)
		work.SetRe(0, float64(lwkopt))
	}
	//
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETRF_AA"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return
	if (*n) == 0 {
		return
	}
	(*ipiv)[0] = 1
	if (*n) == 1 {
		a.Set(0, 0, a.GetReCmplx(0, 0))
		return
	}

	//     Adjust block size based on the workspace size
	if (*lwork) < ((1 + nb) * (*n)) {
		nb = ((*lwork) - (*n)) / (*n)
	}

	if upper {
		//        .....................................................
		//        Factorize A as U**H*D*U using the upper triangle of A
		//        .....................................................
		//
		//        copy first row A(1, 1:N) into H(1:n) (stored in WORK(1:N))
		goblas.Zcopy(*n, a.CVector(0, 0, *lda), work.Off(0, 1))

		//        J is the main loop index, increasing from 1 to N in steps of
		//        JB, where JB is the number of columns factorized by ZLAHEF;
		//        JB is either NB, or N-J+1 for the last block
		j = 0
	label10:
		;
		if j >= (*n) {
			return
		}

		//        each step of the main loop
		//         J is the last column of the previous panel
		//         J1 is the first column of the current panel
		//         K1 identifies if the previous column of the panel has been
		//          explicitly stored, e.g., K1=1 for the first panel, and
		//          K1=0 for the rest
		j1 = j + 1
		jb = min((*n)-j1+1, nb)
		k1 = max(1, j) - j

		//        Panel factorization
		Zlahefaa(uplo, toPtr(2-k1), toPtr((*n)-j), &jb, a.Off(max(1, j)-1, j), lda, toSlice(ipiv, j), work.CMatrix(*n, opts), n, work.Off((*n)*nb))

		//        Adjust IPIV and apply it back (J-th step picks (J+1)-th pivot)
		for j2 = j + 2; j2 <= min(*n, j+jb+1); j2++ {
			(*ipiv)[j2-1] = (*ipiv)[j2-1] + j
			if (j2 != (*ipiv)[j2-1]) && ((j1 - k1) > 2) {
				goblas.Zswap(j1-k1-2, a.CVector(0, j2-1, 1), a.CVector(0, (*ipiv)[j2-1]-1, 1))
			}
		}
		j = j + jb

		//        Trailing submatrix update, where
		//         the row A(J1-1, J2-1:N) stores U(J1, J2+1:N) and
		//         WORK stores the current block of the auxiriarly matrix H
		if j < (*n) {
			//          if the first panel and JB=1 (NB=1), then nothing to do
			if j1 > 1 || jb > 1 {
				//              Merge rank-1 update with BLAS-3 update
				alpha = a.GetConj(j-1, j)
				a.Set(j-1, j, one)
				goblas.Zcopy((*n)-j, a.CVector(j-1-1, j, *lda), work.Off((j+1-j1+1)+jb*(*n)-1, 1))
				goblas.Zscal((*n)-j, alpha, work.Off((j+1-j1+1)+jb*(*n)-1, 1))

				//              K1 identifies if the previous column of the panel has been
				//               explicitly stored, e.g., K1=0 and K2=1 for the first panel,
				//               and K1=1 and K2=0 for the rest
				if j1 > 1 {
					//                 Not first panel
					k2 = 1
				} else {
					//                 First panel
					k2 = 0

					//                 First update skips the first column
					jb = jb - 1
				}

				for j2 = j + 1; j2 <= (*n); j2 += nb {
					nj = min(nb, (*n)-j2+1)

					//                 Update (J2, J2) diagonal block with ZGEMV
					j3 = j2
					for mj = nj - 1; mj >= 1; mj-- {
						err = goblas.Zgemm(ConjTrans, Trans, 1, mj, jb+1, -one, a.Off(j1-k2-1, j3-1), work.CMatrixOff((j3-j1+1)+k1*(*n)-1, *n, opts), one, a.Off(j3-1, j3-1))
						j3 = j3 + 1
					}

					//                 Update off-diagonal block of J2-th block row with ZGEMM
					err = goblas.Zgemm(ConjTrans, Trans, nj, (*n)-j3+1, jb+1, -one, a.Off(j1-k2-1, j2-1), work.CMatrixOff((j3-j1+1)+k1*(*n)-1, *n, opts), one, a.Off(j2-1, j3-1))
				}

				//              Recover T( J, J+1 )
				a.Set(j-1, j, cmplx.Conj(alpha))
			}

			//           WORK(J+1, 1) stores H(J+1, 1)
			goblas.Zcopy((*n)-j, a.CVector(j, j, *lda), work.Off(0, 1))
		}
		goto label10
	} else {
		//        .....................................................
		//        Factorize A as L*D*L**H using the lower triangle of A
		//        .....................................................
		//
		//        copy first column A(1:N, 1) into H(1:N, 1)
		//         (stored in WORK(1:N))
		goblas.Zcopy(*n, a.CVector(0, 0, 1), work.Off(0, 1))

		//        J is the main loop index, increasing from 1 to N in steps of
		//        JB, where JB is the number of columns factorized by ZLAHEF;
		//        JB is either NB, or N-J+1 for the last block
		j = 0
	label11:
		;
		if j >= (*n) {
			return
		}

		//        each step of the main loop
		//         J is the last column of the previous panel
		//         J1 is the first column of the current panel
		//         K1 identifies if the previous column of the panel has been
		//          explicitly stored, e.g., K1=1 for the first panel, and
		//          K1=0 for the rest
		j1 = j + 1
		jb = min((*n)-j1+1, nb)
		k1 = max(1, j) - j

		//        Panel factorization
		Zlahefaa(uplo, toPtr(2-k1), toPtr((*n)-j), &jb, a.Off(j, max(1, j)-1), lda, toSlice(ipiv, j), work.CMatrix(*n, opts), n, work.Off((*n)*nb))

		//        Adjust IPIV and apply it back (J-th step picks (J+1)-th pivot)
		for j2 = j + 2; j2 <= min(*n, j+jb+1); j2++ {
			(*ipiv)[j2-1] = (*ipiv)[j2-1] + j
			if (j2 != (*ipiv)[j2-1]) && ((j1 - k1) > 2) {
				goblas.Zswap(j1-k1-2, a.CVector(j2-1, 0, *lda), a.CVector((*ipiv)[j2-1]-1, 0, *lda))
			}
		}
		j = j + jb

		//        Trailing submatrix update, where
		//          A(J2+1, J1-1) stores L(J2+1, J1) and
		//          WORK(J2+1, 1) stores H(J2+1, 1)
		if j < (*n) {
			//          if the first panel and JB=1 (NB=1), then nothing to do
			if j1 > 1 || jb > 1 {
				//              Merge rank-1 update with BLAS-3 update
				alpha = a.GetConj(j, j-1)
				a.Set(j, j-1, one)
				goblas.Zcopy((*n)-j, a.CVector(j, j-1-1, 1), work.Off((j+1-j1+1)+jb*(*n)-1, 1))
				goblas.Zscal((*n)-j, alpha, work.Off((j+1-j1+1)+jb*(*n)-1, 1))

				//              K1 identifies if the previous column of the panel has been
				//               explicitly stored, e.g., K1=0 and K2=1 for the first panel,
				//               and K1=1 and K2=0 for the rest
				if j1 > 1 {
					//                 Not first panel
					k2 = 1
				} else {
					//                 First panel
					k2 = 0

					//                 First update skips the first column
					jb = jb - 1
				}

				for j2 = j + 1; j2 <= (*n); j2 += nb {
					nj = min(nb, (*n)-j2+1)

					//                 Update (J2, J2) diagonal block with ZGEMV
					j3 = j2
					for mj = nj - 1; mj >= 1; mj-- {
						err = goblas.Zgemm(NoTrans, ConjTrans, mj, 1, jb+1, -one, work.CMatrixOff((j3-j1+1)+k1*(*n)-1, *n, opts), a.Off(j3-1, j1-k2-1), one, a.Off(j3-1, j3-1))
						j3 = j3 + 1
					}

					//                 Update off-diagonal block of J2-th block column with ZGEMM
					err = goblas.Zgemm(NoTrans, ConjTrans, (*n)-j3+1, nj, jb+1, -one, work.CMatrixOff((j3-j1+1)+k1*(*n)-1, *n, opts), a.Off(j2-1, j1-k2-1), one, a.Off(j3-1, j2-1))
				}

				//              Recover T( J+1, J )
				a.Set(j, j-1, cmplx.Conj(alpha))
			}

			//           WORK(J+1, 1) stores H(J+1, 1)
			goblas.Zcopy((*n)-j, a.CVector(j, j, 1), work.Off(0, 1))
		}
		goto label11
	}
}
