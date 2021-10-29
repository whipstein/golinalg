package golapack

import "github.com/whipstein/golinalg/mat"

// Dlapmt rearranges the columns of the M by N matrix X as specified
// by the permutation K(1),K(2),...,K(N) of the integers 1,...,N.
// If FORWRD = .TRUE.,  forward permutation:
//
//      X(*,K(J)) is moved X(*,J) for J = 1,2,...,N.
//
// If FORWRD = .FALSE., backward permutation:
//
//      X(*,J) is moved to X(*,K(J)) for J = 1,2,...,N.
func Dlapmt(forwrd bool, m, n int, x *mat.Matrix, k *[]int) {
	var temp float64
	var i, ii, in, j int

	if n <= 1 {
		return
	}

	for i = 1; i <= n; i++ {
		(*k)[i-1] = -(*k)[i-1]
	}

	if forwrd {
		//        Forward permutation
		for i = 1; i <= n; i++ {

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

			for ii = 1; ii <= m; ii++ {
				temp = x.Get(ii-1, j-1)
				x.Set(ii-1, j-1, x.Get(ii-1, in-1))
				x.Set(ii-1, in-1, temp)
			}

			(*k)[in-1] = -(*k)[in-1]
			j = in
			in = (*k)[in-1]
			goto label20

		label40:
		}

	} else {
		//        Backward permutation
		for i = 1; i <= n; i++ {

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

			for ii = 1; ii <= m; ii++ {
				temp = x.Get(ii-1, i-1)
				x.Set(ii-1, i-1, x.Get(ii-1, j-1))
				x.Set(ii-1, j-1, temp)
			}

			(*k)[j-1] = -(*k)[j-1]
			j = (*k)[j-1]
			goto label60

		label80:
		}

	}
}
