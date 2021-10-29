package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrf computes the factorization of a complex Hermitian matrix A
// using the Bunch-Kaufman diagonal pivoting method.  The form of the
// factorization is
//
//    A = U*D*U**H  or  A = L*D*L**H
//
// where U (or L) is a product of permutation and unit upper (lower)
// triangular matrices, and D is Hermitian and block diagonal with
// 1-by-1 and 2-by-2 diagonal blocks.
//
// This is the blocked version of the algorithm, calling Level 3 BLAS.
func Zhetrf(uplo mat.MatUplo, n int, a *mat.CMatrix, ipiv *[]int, work *mat.CVector, lwork int) (info int, err error) {
	var lquery, upper bool
	var iinfo, iws, j, k, kb, ldwork, lwkopt, nb, nbmin int

	//     Test the input parameters.
	upper = uplo == Upper
	lquery = (lwork == -1)
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if lwork < 1 && !lquery {
		err = fmt.Errorf("lwork < 1 && !lquery: lwork=%v, lquery=%v", lwork, lquery)
	}

	if err == nil {
		//        Determine the block size
		nb = Ilaenv(1, "Zhetrf", []byte{uplo.Byte()}, n, -1, -1, -1)
		lwkopt = n * nb
		work.SetRe(0, float64(lwkopt))
	}
	//
	if err != nil {
		gltest.Xerbla2("Zhetrf", err)
		return
	} else if lquery {
		return
	}

	nbmin = 2
	ldwork = n
	if nb > 1 && nb < n {
		iws = ldwork * nb
		if lwork < iws {
			nb = max(lwork/ldwork, 1)
			nbmin = max(2, Ilaenv(2, "Zhetrf", []byte{uplo.Byte()}, n, -1, -1, -1))
		}
	} else {
		iws = 1
	}
	if nb < nbmin {
		nb = n
	}

	if upper {
		//        Factorize A as U*D*U**H using the upper triangle of A
		//
		//        K is the main loop index, decreasing from N to 1 in steps of
		//        KB, where KB is the number of columns factorized by ZLAHEF;
		//        KB is either NB or NB-1, or K for the last block
		k = n
	label10:
		;

		//        If K < 1, exit from loop
		if k < 1 {
			goto label40
		}

		if k > nb {
			//           Factorize columns k-kb+1:k of A and use blocked code to
			//           update columns 1:k-kb
			kb, iinfo = Zlahef(uplo, k, nb, a, ipiv, work.CMatrix(n, opts))
		} else {
			//           Use unblocked code to factorize columns 1:k of A
			if iinfo, err = Zhetf2(uplo, k, a, ipiv); err != nil {
				panic(err)
			}
			kb = k
		}

		//        Set INFO on the first occurrence of a zero pivot
		if info == 0 && iinfo > 0 {
			info = iinfo
		}

		//        Decrease K and return to the start of the main loop
		k = k - kb
		goto label10

	} else {
		//        Factorize A as L*D*L**H using the lower triangle of A
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        KB, where KB is the number of columns factorized by ZLAHEF;
		//        KB is either NB or NB-1, or N-K+1 for the last block
		k = 1
	label20:
		;

		//        If K > N, exit from loop
		if k > n {
			goto label40
		}

		if k <= n-nb {
			//           Factorize columns k:k+kb-1 of A and use blocked code to
			//           update columns k+kb:n
			kb, iinfo = Zlahef(uplo, n-k+1, nb, a.Off(k-1, k-1), toSlice(ipiv, k-1), work.CMatrix(n, opts))
		} else {
			//           Use unblocked code to factorize columns k:n of A
			if iinfo, err = Zhetf2(uplo, n-k+1, a.Off(k-1, k-1), toSlice(ipiv, k-1)); err != nil {
				panic(err)
			}
			kb = n - k + 1
		}

		//        Set INFO on the first occurrence of a zero pivot
		if info == 0 && iinfo > 0 {
			info = iinfo + k - 1
		}

		//        Adjust IPIV
		for j = k; j <= k+kb-1; j++ {
			if (*ipiv)[j-1] > 0 {
				(*ipiv)[j-1] = (*ipiv)[j-1] + k - 1
			} else {
				(*ipiv)[j-1] = (*ipiv)[j-1] - k + 1
			}
		}

		//        Increase K and return to the start of the main loop
		k = k + kb
		goto label20

	}

label40:
	;
	work.SetRe(0, float64(lwkopt))

	return
}
