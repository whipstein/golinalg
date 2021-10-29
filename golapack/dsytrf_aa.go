package golapack

import (
	"fmt"

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
func DsytrfAa(uplo mat.MatUplo, n int, a *mat.Matrix, ipiv *[]int, work *mat.Vector, lwork int) (info int, err error) {
	var lquery, upper bool
	var alpha, one float64
	var j, j1, j2, j3, jb, k1, k2, lwkopt, mj, nb, nj int

	one = 1.0

	//     Determine the block size
	nb = Ilaenv(1, "DsytrfAa", []byte{uplo.Byte()}, n, -1, -1, -1)

	//     Test the input parameters.
	upper = uplo == Upper
	lquery = (lwork == -1)
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if lwork < max(1, 2*n) && !lquery {
		err = fmt.Errorf("lwork < max(1, 2*n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}

	if err == nil {
		lwkopt = (nb + 1) * n
		work.Set(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("DsytrfAa", err)
		return
	} else if lquery {
		return
	}

	//     Quick return
	if n == 0 {
		return
	}
	(*ipiv)[0] = 1
	if n == 1 {
		return
	}

	//     Adjust block size based on the workspace size
	if lwork < ((1 + nb) * n) {
		nb = (lwork - n) / n
	}

	if upper {
		//        .....................................................
		//        Factorize A as U**T*D*U using the upper triangle of A
		//        .....................................................
		//
		//        Copy first row A(1, 1:N) into H(1:n) (stored in WORK(1:N))
		goblas.Dcopy(n, a.Vector(0, 0), work.Off(0, 1))

		//        J is the main loop index, increasing from 1 to N in steps of
		//        JB, where JB is the number of columns factorized by DLASYF;
		//        JB is either NB, or N-J+1 for the last block
		j = 0
	label10:
		;
		if j >= n {
			return
		}

		//        each step of the main loop
		//         J is the last column of the previous panel
		//         J1 is the first column of the current panel
		//         K1 identifies if the previous column of the panel has been
		//          explicitly stored, e.g., K1=1 for the first panel, and
		//          K1=0 for the rest
		j1 = j + 1
		jb = min(n-j1+1, nb)
		k1 = max(1, j) - j

		//        Panel factorization
		DlasyfAa(uplo, 2-k1, n-j, jb, a.Off(max(1, j)-1, j), toSlice(ipiv, j), work.Matrix(n, opts), work.Off(n*nb))

		//        Adjust IPIV and apply it back (J-th step picks (J+1)-th pivot)
		for j2 = j + 2; j2 <= min(n, j+jb+1); j2++ {
			(*ipiv)[j2-1] = (*ipiv)[j2-1] + j
			if (j2 != (*ipiv)[j2-1]) && ((j1 - k1) > 2) {
				goblas.Dswap(j1-k1-2, a.Vector(0, j2-1, 1), a.Vector(0, (*ipiv)[j2-1]-1, 1))
			}
		}
		j = j + jb

		//        Trailing submatrix update, where
		//         the row A(J1-1, J2-1:N) stores U(J1, J2+1:N) and
		//         WORK stores the current block of the auxiriarly matrix H
		if j < n {
			//           If first panel and JB=1 (NB=1), then nothing to do
			if j1 > 1 || jb > 1 {
				//              Merge rank-1 update with BLAS-3 update
				alpha = a.Get(j-1, j)
				a.Set(j-1, j, one)
				goblas.Dcopy(n-j, a.Vector(j-1-1, j), work.Off((j+1-j1+1)+jb*n-1, 1))
				goblas.Dscal(n-j, alpha, work.Off((j+1-j1+1)+jb*n-1, 1))

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

				for j2 = j + 1; j2 <= n; j2 += nb {
					nj = min(nb, n-j2+1)

					//                 Update (J2, J2) diagonal block with DGEMV
					j3 = j2
					for mj = nj - 1; mj >= 1; mj-- {
						if err = goblas.Dgemv(NoTrans, mj, jb+1, -one, work.MatrixOff(j3-j1+1+k1*n-1, n, opts), a.Vector(j1-k2-1, j3-1, 1), one, a.Vector(j3-1, j3-1)); err != nil {
							panic(err)
						}
						j3 = j3 + 1
					}

					//                 Update off-diagonal block of J2-th block row with DGEMM
					if err = goblas.Dgemm(Trans, Trans, nj, n-j3+1, jb+1, -one, a.Off(j1-k2-1, j2-1), work.MatrixOff(j3-j1+1+k1*n-1, n, opts), one, a.Off(j2-1, j3-1)); err != nil {
						panic(err)
					}
				}

				//              Recover T( J, J+1 )
				a.Set(j-1, j, alpha)
			}

			//           WORK(J+1, 1) stores H(J+1, 1)
			goblas.Dcopy(n-j, a.Vector(j, j), work.Off(0, 1))
		}
		goto label10
	} else {
		//        .....................................................
		//        Factorize A as L*D*L**T using the lower triangle of A
		//        .....................................................
		//
		//        copy first column A(1:N, 1) into H(1:N, 1)
		//         (stored in WORK(1:N))
		goblas.Dcopy(n, a.Vector(0, 0, 1), work.Off(0, 1))

		//        J is the main loop index, increasing from 1 to N in steps of
		//        JB, where JB is the number of columns factorized by DLASYF;
		//        JB is either NB, or N-J+1 for the last block
		j = 0
	label11:
		;
		if j >= n {
			return
		}

		//        each step of the main loop
		//         J is the last column of the previous panel
		//         J1 is the first column of the current panel
		//         K1 identifies if the previous column of the panel has been
		//          explicitly stored, e.g., K1=1 for the first panel, and
		//          K1=0 for the rest
		j1 = j + 1
		jb = min(n-j1+1, nb)
		k1 = max(1, j) - j

		//        Panel factorization
		DlasyfAa(uplo, 2-k1, n-j, jb, a.Off(j, max(1, j)-1), toSlice(ipiv, j), work.Matrix(n, opts), work.Off(n*nb))

		//        Adjust IPIV and apply it back (J-th step picks (J+1)-th pivot)
		for j2 = j + 2; j2 <= min(n, j+jb+1); j2++ {
			(*ipiv)[j2-1] = (*ipiv)[j2-1] + j
			if (j2 != (*ipiv)[j2-1]) && ((j1 - k1) > 2) {
				goblas.Dswap(j1-k1-2, a.Vector(j2-1, 0), a.Vector((*ipiv)[j2-1]-1, 0))
			}
		}
		j = j + jb

		//        Trailing submatrix update, where
		//          A(J2+1, J1-1) stores L(J2+1, J1) and
		//          WORK(J2+1, 1) stores H(J2+1, 1)
		if j < n {
			//           if first panel and JB=1 (NB=1), then nothing to do
			if j1 > 1 || jb > 1 {
				//              Merge rank-1 update with BLAS-3 update
				alpha = a.Get(j, j-1)
				a.Set(j, j-1, one)
				goblas.Dcopy(n-j, a.Vector(j, j-1-1, 1), work.Off((j+1-j1+1)+jb*n-1, 1))
				goblas.Dscal(n-j, alpha, work.Off((j+1-j1+1)+jb*n-1, 1))

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

				for j2 = j + 1; j2 <= n; j2 += nb {
					nj = min(nb, n-j2+1)

					//                 Update (J2, J2) diagonal block with DGEMV
					j3 = j2
					for mj = nj - 1; mj >= 1; mj-- {
						if err = goblas.Dgemv(NoTrans, mj, jb+1, -one, work.MatrixOff(j3-j1+1+k1*n-1, n, opts), a.Vector(j3-1, j1-k2-1), one, a.Vector(j3-1, j3-1, 1)); err != nil {
							panic(err)
						}
						j3 = j3 + 1
					}

					//                 Update off-diagonal block in J2-th block column with DGEMM
					if err = goblas.Dgemm(NoTrans, Trans, n-j3+1, nj, jb+1, -one, work.MatrixOff(j3-j1+1+k1*n-1, n, opts), a.Off(j2-1, j1-k2-1), one, a.Off(j3-1, j2-1)); err != nil {
						panic(err)
					}
				}

				//              Recover T( J+1, J )
				a.Set(j, j-1, alpha)
			}

			//           WORK(J+1, 1) stores H(J+1, 1)
			goblas.Dcopy(n-j, a.Vector(j, j, 1), work.Off(0, 1))
		}
		goto label11
	}
}
