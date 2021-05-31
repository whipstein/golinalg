package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DsytrfAa computes the factorization of a real symmetric matrix A
// using the Aasen's algorithm.  The form of the factorization is
//
//    A = U**T*T*U  or  A = L*T*L**T
//
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and T is a symmetric tridiagonal matrix.
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
func DsytrfAa(uplo byte, n *int, a *mat.Matrix, lda *int, ipiv *[]int, work *mat.Vector, lwork, info *int) {
	var lquery, upper bool
	var alpha, one float64
	var j, j1, j2, j3, jb, k1, k2, lwkopt, mj, nb, nj int

	one = 1.0

	//     Determine the block size
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSYTRF_AA"), []byte{uplo}, n, toPtr(-1), toPtr(-1), toPtr(-1))

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	lquery = ((*lwork) == -1)
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	} else if (*lwork) < maxint(1, 2*(*n)) && !lquery {
		(*info) = -7
	}

	if (*info) == 0 {
		lwkopt = (nb + 1) * (*n)
		work.Set(0, float64(lwkopt))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTRF_AA"), -(*info))
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
		return
	}

	//     Adjust block size based on the workspace size
	if (*lwork) < ((1 + nb) * (*n)) {
		nb = ((*lwork) - (*n)) / (*n)
	}

	if upper {
		//        .....................................................
		//        Factorize A as U**T*D*U using the upper triangle of A
		//        .....................................................
		//
		//        Copy first row A(1, 1:N) into H(1:n) (stored in WORK(1:N))
		goblas.Dcopy(n, a.Vector(0, 0), lda, work, func() *int { y := 1; return &y }())

		//        J is the main loop index, increasing from 1 to N in steps of
		//        JB, where JB is the number of columns factorized by DLASYF;
		//        JB is either NB, or N-J+1 for the last block
		j = 0
	label10:
		;
		if j >= (*n) {
			goto label20
		}

		//        each step of the main loop
		//         J is the last column of the previous panel
		//         J1 is the first column of the current panel
		//         K1 identifies if the previous column of the panel has been
		//          explicitly stored, e.g., K1=1 for the first panel, and
		//          K1=0 for the rest
		j1 = j + 1
		jb = minint((*n)-j1+1, nb)
		k1 = maxint(1, j) - j

		//        Panel factorization
		DlasyfAa(uplo, toPtr(2-k1), toPtr((*n)-j), &jb, a.Off(maxint(1, j)-1, j+1-1), lda, toSlice(ipiv, j+1-1), work.Matrix(*n, opts), n, work.Off((*n)*nb+1-1))

		//        Adjust IPIV and apply it back (J-th step picks (J+1)-th pivot)
		for j2 = j + 2; j2 <= minint(*n, j+jb+1); j2++ {
			(*ipiv)[j2-1] = (*ipiv)[j2-1] + j
			if (j2 != (*ipiv)[j2-1]) && ((j1 - k1) > 2) {
				goblas.Dswap(toPtr(j1-k1-2), a.Vector(0, j2-1), func() *int { y := 1; return &y }(), a.Vector(0, (*ipiv)[j2-1]-1), func() *int { y := 1; return &y }())
			}
		}
		j = j + jb

		//        Trailing submatrix update, where
		//         the row A(J1-1, J2-1:N) stores U(J1, J2+1:N) and
		//         WORK stores the current block of the auxiriarly matrix H
		if j < (*n) {
			//           If first panel and JB=1 (NB=1), then nothing to do
			if j1 > 1 || jb > 1 {
				//              Merge rank-1 update with BLAS-3 update
				alpha = a.Get(j-1, j+1-1)
				a.Set(j-1, j+1-1, one)
				goblas.Dcopy(toPtr((*n)-j), a.Vector(j-1-1, j+1-1), lda, work.Off((j+1-j1+1)+jb*(*n)-1), func() *int { y := 1; return &y }())
				goblas.Dscal(toPtr((*n)-j), &alpha, work.Off((j+1-j1+1)+jb*(*n)-1), func() *int { y := 1; return &y }())

				//              K1 identifies if the previous column of the panel has been
				//               explicitly stored, e.g., K1=1 and K2= 0 for the first panel,
				//               while K1=0 and K2=1 for the rest
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
					nj = minint(nb, (*n)-j2+1)

					//                 Update (J2, J2) diagonal block with DGEMV
					j3 = j2
					for mj = nj - 1; mj >= 1; mj-- {
						goblas.Dgemv(NoTrans, &mj, toPtr(jb+1), toPtrf64(-one), work.MatrixOff(j3-j1+1+k1*(*n)-1, *n, opts), n, a.Vector(j1-k2-1, j3-1), func() *int { y := 1; return &y }(), &one, a.Vector(j3-1, j3-1), lda)
						j3 = j3 + 1
					}

					//                 Update off-diagonal block of J2-th block row with DGEMM
					goblas.Dgemm(Trans, Trans, &nj, toPtr((*n)-j3+1), toPtr(jb+1), toPtrf64(-one), a.Off(j1-k2-1, j2-1), lda, work.MatrixOff(j3-j1+1+k1*(*n)-1, *n, opts), n, &one, a.Off(j2-1, j3-1), lda)
				}

				//              Recover T( J, J+1 )
				a.Set(j-1, j+1-1, alpha)
			}

			//           WORK(J+1, 1) stores H(J+1, 1)
			goblas.Dcopy(toPtr((*n)-j), a.Vector(j+1-1, j+1-1), lda, work, func() *int { y := 1; return &y }())
		}
		goto label10
	} else {
		//        .....................................................
		//        Factorize A as L*D*L**T using the lower triangle of A
		//        .....................................................
		//
		//        copy first column A(1:N, 1) into H(1:N, 1)
		//         (stored in WORK(1:N))
		goblas.Dcopy(n, a.Vector(0, 0), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())

		//        J is the main loop index, increasing from 1 to N in steps of
		//        JB, where JB is the number of columns factorized by DLASYF;
		//        JB is either NB, or N-J+1 for the last block
		j = 0
	label11:
		;
		if j >= (*n) {
			goto label20
		}

		//        each step of the main loop
		//         J is the last column of the previous panel
		//         J1 is the first column of the current panel
		//         K1 identifies if the previous column of the panel has been
		//          explicitly stored, e.g., K1=1 for the first panel, and
		//          K1=0 for the rest
		j1 = j + 1
		jb = minint((*n)-j1+1, nb)
		k1 = maxint(1, j) - j

		//        Panel factorization
		DlasyfAa(uplo, toPtr(2-k1), toPtr((*n)-j), &jb, a.Off(j+1-1, maxint(1, j)-1), lda, toSlice(ipiv, j+1-1), work.Matrix(*n, opts), n, work.Off((*n)*nb+1-1))

		//        Adjust IPIV and apply it back (J-th step picks (J+1)-th pivot)
		for j2 = j + 2; j2 <= minint(*n, j+jb+1); j2++ {
			(*ipiv)[j2-1] = (*ipiv)[j2-1] + j
			if (j2 != (*ipiv)[j2-1]) && ((j1 - k1) > 2) {
				goblas.Dswap(toPtr(j1-k1-2), a.Vector(j2-1, 0), lda, a.Vector((*ipiv)[j2-1]-1, 0), lda)
			}
		}
		j = j + jb

		//        Trailing submatrix update, where
		//          A(J2+1, J1-1) stores L(J2+1, J1) and
		//          WORK(J2+1, 1) stores H(J2+1, 1)
		if j < (*n) {
			//           if first panel and JB=1 (NB=1), then nothing to do
			if j1 > 1 || jb > 1 {
				//              Merge rank-1 update with BLAS-3 update
				alpha = a.Get(j+1-1, j-1)
				a.Set(j+1-1, j-1, one)
				goblas.Dcopy(toPtr((*n)-j), a.Vector(j+1-1, j-1-1), func() *int { y := 1; return &y }(), work.Off((j+1-j1+1)+jb*(*n)-1), func() *int { y := 1; return &y }())
				goblas.Dscal(toPtr((*n)-j), &alpha, work.Off((j+1-j1+1)+jb*(*n)-1), func() *int { y := 1; return &y }())

				//              K1 identifies if the previous column of the panel has been
				//               explicitly stored, e.g., K1=1 and K2= 0 for the first panel,
				//               while K1=0 and K2=1 for the rest
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
					nj = minint(nb, (*n)-j2+1)

					//                 Update (J2, J2) diagonal block with DGEMV
					j3 = j2
					for mj = nj - 1; mj >= 1; mj-- {
						goblas.Dgemv(NoTrans, &mj, toPtr(jb+1), toPtrf64(-one), work.MatrixOff(j3-j1+1+k1*(*n)-1, *n, opts), n, a.Vector(j3-1, j1-k2-1), lda, &one, a.Vector(j3-1, j3-1), func() *int { y := 1; return &y }())
						j3 = j3 + 1
					}

					//                 Update off-diagonal block in J2-th block column with DGEMM
					goblas.Dgemm(NoTrans, Trans, toPtr((*n)-j3+1), &nj, toPtr(jb+1), toPtrf64(-one), work.MatrixOff(j3-j1+1+k1*(*n)-1, *n, opts), n, a.Off(j2-1, j1-k2-1), lda, &one, a.Off(j3-1, j2-1), lda)
				}

				//              Recover T( J+1, J )
				a.Set(j+1-1, j-1, alpha)
			}

			//           WORK(J+1, 1) stores H(J+1, 1)
			goblas.Dcopy(toPtr((*n)-j), a.Vector(j+1-1, j+1-1), func() *int { y := 1; return &y }(), work.Off(0), func() *int { y := 1; return &y }())
		}
		goto label11
	}

label20:
}
