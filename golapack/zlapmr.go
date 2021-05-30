package golapack

import "golinalg/mat"

// Zlapmr rearranges the rows of the M by N matrix X as specified
// by the permutation K(1),K(2),...,K(M) of the integers 1,...,M.
// If FORWRD = .TRUE.,  forward permutation:
//
//      X(K(I),*) is moved X(I,*) for I = 1,2,...,M.
//
// If FORWRD = .FALSE., backward permutation:
//
//      X(I,*) is moved to X(K(I),*) for I = 1,2,...,M.
func Zlapmr(forwrd bool, m, n *int, x *mat.CMatrix, ldx *int, k *[]int) {
	var temp complex128
	var i, in, j, jj int

	if (*m) <= 1 {
		return
	}
	//
	for i = 1; i <= (*m); i++ {
		(*k)[i-1] = -(*k)[i-1]
	}

	if forwrd {
		//        Forward permutation
		for i = 1; i <= (*m); i++ {

			if (*k)[i-1] > 0 {
				goto label40
			}

			j = i
			(*k)[j-1] = -(*k)[j-1]
			in = (*k)[j-1]

		label20:
			;
			if (*k)[in-1] > 0 {
				goto label40
			}

			for jj = 1; jj <= (*n); jj++ {
				temp = x.Get(j-1, jj-1)
				x.Set(j-1, jj-1, x.Get(in-1, jj-1))
				x.Set(in-1, jj-1, temp)
			}

			(*k)[in-1] = -(*k)[in-1]
			j = in
			in = (*k)[in-1]
			goto label20

		label40:
		}

	} else {
		//        Backward permutation
		for i = 1; i <= (*m); i++ {

			if (*k)[i-1] > 0 {
				goto label80
			}

			(*k)[i-1] = -(*k)[i-1]
			j = (*k)[i-1]
		label60:
			;
			if j == i {
				goto label80
			}

			for jj = 1; jj <= (*n); jj++ {
				temp = x.Get(i-1, jj-1)
				x.Set(i-1, jj-1, x.Get(j-1, jj-1))
				x.Set(j-1, jj-1, temp)
			}

			(*k)[j-1] = -(*k)[j-1]
			j = (*k)[j-1]
			goto label60

		label80:
		}

	}
}
