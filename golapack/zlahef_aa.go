package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Zlahefaa factorizes a panel of a complex hermitian matrix A using
// the Aasen's algorithm. The panel consists of a set of NB rows of A
// when UPLO is U, or a set of NB columns when UPLO is L.
//
// In order to factorize the panel, the Aasen's algorithm requires the
// last row, or column, of the previous panel. The first row, or column,
// of A is set to be the first row, or column, of an identity matrix,
// which is used to factorize the first panel.
//
// The resulting J-th row of U, or J-th column of L, is stored in the
// (J-1)-th row, or column, of A (without the unit diagonals), while
// the diagonal and subdiagonal of A are overwritten by those of T.
func Zlahefaa(uplo byte, j1, m, nb *int, a *mat.CMatrix, lda *int, ipiv *[]int, h *mat.CMatrix, ldh *int, work *mat.CVector) {
	var alpha, one, piv, zero complex128
	var i1, i2, j, k, k1, mj int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	j = 1

	//     K1 is the first column of the panel to be factorized
	//     i.e.,  K1 is 2 for the first block column, and 1 for the rest of the blocks
	k1 = (2 - (*j1)) + 1

	if uplo == 'U' {
		//        .....................................................
		//        Factorize A as U**T*D*U using the upper triangle of A
		//        .....................................................
	label10:
		;
		if j > minint(*m, *nb) {
			goto label20
		}

		//        K is the column to be factorized
		//         when being called from ZHETRF_AA,
		//         > for the first block column, J1 is 1, hence J1+J-1 is J,
		//         > for the rest of the columns, J1 is 2, and J1+J-1 is J+1,
		k = (*j1) + j - 1
		if j == (*m) {
			//            Only need to compute T(J, J)
			mj = 1
		} else {
			mj = (*m) - j + 1
		}

		//        H(J:N, J) := A(J, J:N) - H(J:N, 1:(J-1)) * L(J1:(J-1), J),
		//         where H(J:N, J) has been initialized to be A(J, J:N)
		if k > 2 {
			//        K is the column to be factorized
			//         > for the first block column, K is J, skipping the first two
			//           columns
			//         > for the rest of the columns, K is J+1, skipping only the
			//           first column
			Zlacgv(toPtr(j-k1), a.CVector(0, j-1), func() *int { y := 1; return &y }())
			goblas.Zgemv(NoTrans, &mj, toPtr(j-k1), toPtrc128(-one), h.Off(j-1, k1-1), ldh, a.CVector(0, j-1), func() *int { y := 1; return &y }(), &one, h.CVector(j-1, j-1), func() *int { y := 1; return &y }())
			Zlacgv(toPtr(j-k1), a.CVector(0, j-1), func() *int { y := 1; return &y }())
		}
		//        Copy H(i:n, i) into WORK
		goblas.Zcopy(&mj, h.CVector(j-1, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())

		if j > k1 {
			//           Compute WORK := WORK - L(J-1, J:N) * T(J-1,J),
			//            where A(J-1, J) stores T(J-1, J) and A(J-2, J:N) stores U(J-1, J:N)
			alpha = -a.GetConj(k-1-1, j-1)
			goblas.Zaxpy(&mj, &alpha, a.CVector(k-2-1, j-1), lda, work, func() *int { y := 1; return &y }())
		}

		//        Set A(J, J) = T(J, J)
		a.Set(k-1, j-1, work.GetReCmplx(0))

		if j < (*m) {
			//           Compute WORK(2:N) = T(J, J) L(J, (J+1):N)
			//            where A(J, J) stores T(J, J) and A(J-1, (J+1):N) stores U(J, (J+1):N)
			if k > 1 {
				alpha = -a.Get(k-1, j-1)
				goblas.Zaxpy(toPtr((*m)-j), &alpha, a.CVector(k-1-1, j+1-1), lda, work.Off(1), func() *int { y := 1; return &y }())
			}

			//           Find max(|WORK(2:n)|)
			i2 = goblas.Izamax(toPtr((*m)-j), work.Off(1), func() *int { y := 1; return &y }()) + 1
			piv = work.Get(i2 - 1)

			//           Apply hermitian pivot
			if (i2 != 2) && (piv != 0) {
				//              Swap WORK(I1) and WORK(I2)
				i1 = 2
				work.Set(i2-1, work.Get(i1-1))
				work.Set(i1-1, piv)

				//              Swap A(I1, I1+1:N) with A(I1+1:N, I2)
				i1 = i1 + j - 1
				i2 = i2 + j - 1
				goblas.Zswap(toPtr(i2-i1-1), a.CVector((*j1)+i1-1-1, i1+1-1), lda, a.CVector((*j1)+i1-1, i2-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr(i2-i1), a.CVector((*j1)+i1-1-1, i1+1-1), lda)
				Zlacgv(toPtr(i2-i1-1), a.CVector((*j1)+i1-1, i2-1), func() *int { y := 1; return &y }())

				//              Swap A(I1, I2+1:N) with A(I2, I2+1:N)
				if i2 < (*m) {
					goblas.Zswap(toPtr((*m)-i2), a.CVector((*j1)+i1-1-1, i2+1-1), lda, a.CVector((*j1)+i2-1-1, i2+1-1), lda)
				}

				//              Swap A(I1, I1) with A(I2,I2)
				piv = a.Get(i1+(*j1)-1-1, i1-1)
				a.Set((*j1)+i1-1-1, i1-1, a.Get((*j1)+i2-1-1, i2-1))
				a.Set((*j1)+i2-1-1, i2-1, piv)

				//              Swap H(I1, 1:J1) with H(I2, 1:J1)
				goblas.Zswap(toPtr(i1-1), h.CVector(i1-1, 0), ldh, h.CVector(i2-1, 0), ldh)
				(*ipiv)[i1-1] = i2

				if i1 > (k1 - 1) {
					//                 Swap L(1:I1-1, I1) with L(1:I1-1, I2),
					//                  skipping the first column
					goblas.Zswap(toPtr(i1-k1+1), a.CVector(0, i1-1), func() *int { y := 1; return &y }(), a.CVector(0, i2-1), func() *int { y := 1; return &y }())
				}
			} else {
				(*ipiv)[j+1-1] = j + 1
			}

			//           Set A(J, J+1) = T(J, J+1)
			a.Set(k-1, j+1-1, work.Get(1))

			if j < (*nb) {
				//              Copy A(J+1:N, J+1) into H(J:N, J),
				goblas.Zcopy(toPtr((*m)-j), a.CVector(k+1-1, j+1-1), lda, h.CVector(j+1-1, j+1-1), func() *int { y := 1; return &y }())
			}

			//           Compute L(J+2, J+1) = WORK( 3:N ) / T(J, J+1),
			//            where A(J, J+1) = T(J, J+1) and A(J+2:N, J) = L(J+2:N, J+1)
			if j < ((*m) - 1) {
				if a.Get(k-1, j+1-1) != zero {
					alpha = one / a.Get(k-1, j+1-1)
					goblas.Zcopy(toPtr((*m)-j-1), work.Off(2), func() *int { y := 1; return &y }(), a.CVector(k-1, j+2-1), lda)
					goblas.Zscal(toPtr((*m)-j-1), &alpha, a.CVector(k-1, j+2-1), lda)
				} else {
					Zlaset('F', func() *int { y := 1; return &y }(), toPtr((*m)-j-1), &zero, &zero, a.Off(k-1, j+2-1), lda)
				}
			}
		}
		j = j + 1
		goto label10
	label20:
	} else {
		//        .....................................................
		//        Factorize A as L*D*L**T using the lower triangle of A
		//        .....................................................
	label30:
		;
		if j > minint(*m, *nb) {
			goto label40
		}

		//        K is the column to be factorized
		//         when being called from ZHETRF_AA,
		//         > for the first block column, J1 is 1, hence J1+J-1 is J,
		//         > for the rest of the columns, J1 is 2, and J1+J-1 is J+1,
		k = (*j1) + j - 1
		if j == (*m) {
			//            Only need to compute T(J, J)
			mj = 1
		} else {
			mj = (*m) - j + 1
		}

		//        H(J:N, J) := A(J:N, J) - H(J:N, 1:(J-1)) * L(J, J1:(J-1))^T,
		//         where H(J:N, J) has been initialized to be A(J:N, J)
		if k > 2 {
			//        K is the column to be factorized
			//         > for the first block column, K is J, skipping the first two
			//           columns
			//         > for the rest of the columns, K is J+1, skipping only the
			//           first column
			Zlacgv(toPtr(j-k1), a.CVector(j-1, 0), lda)
			goblas.Zgemv(NoTrans, &mj, toPtr(j-k1), toPtrc128(-one), h.Off(j-1, k1-1), ldh, a.CVector(j-1, 0), lda, &one, h.CVector(j-1, j-1), func() *int { y := 1; return &y }())
			Zlacgv(toPtr(j-k1), a.CVector(j-1, 0), lda)
		}

		//        Copy H(J:N, J) into WORK
		goblas.Zcopy(&mj, h.CVector(j-1, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())

		if j > k1 {
			//           Compute WORK := WORK - L(J:N, J-1) * T(J-1,J),
			//            where A(J-1, J) = T(J-1, J) and A(J, J-2) = L(J, J-1)
			alpha = -a.GetConj(j-1, k-1-1)
			goblas.Zaxpy(&mj, &alpha, a.CVector(j-1, k-2-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
		}

		//        Set A(J, J) = T(J, J)
		a.Set(j-1, k-1, work.GetReCmplx(0))

		if j < (*m) {
			//           Compute WORK(2:N) = T(J, J) L((J+1):N, J)
			//            where A(J, J) = T(J, J) and A((J+1):N, J-1) = L((J+1):N, J)
			if k > 1 {
				alpha = -a.Get(j-1, k-1)
				goblas.Zaxpy(toPtr((*m)-j), &alpha, a.CVector(j+1-1, k-1-1), func() *int { y := 1; return &y }(), work.Off(1), func() *int { y := 1; return &y }())
			}

			//           Find max(|WORK(2:n)|)
			i2 = goblas.Izamax(toPtr((*m)-j), work.Off(1), func() *int { y := 1; return &y }()) + 1
			piv = work.Get(i2 - 1)

			//           Apply hermitian pivot
			if (i2 != 2) && (piv != 0) {
				//              Swap WORK(I1) and WORK(I2)
				i1 = 2
				work.Set(i2-1, work.Get(i1-1))
				work.Set(i1-1, piv)

				//              Swap A(I1+1:N, I1) with A(I2, I1+1:N)
				i1 = i1 + j - 1
				i2 = i2 + j - 1
				goblas.Zswap(toPtr(i2-i1-1), a.CVector(i1+1-1, (*j1)+i1-1-1), func() *int { y := 1; return &y }(), a.CVector(i2-1, (*j1)+i1-1), lda)
				Zlacgv(toPtr(i2-i1), a.CVector(i1+1-1, (*j1)+i1-1-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr(i2-i1-1), a.CVector(i2-1, (*j1)+i1-1), lda)

				//              Swap A(I2+1:N, I1) with A(I2+1:N, I2)
				if i2 < (*m) {
					goblas.Zswap(toPtr((*m)-i2), a.CVector(i2+1-1, (*j1)+i1-1-1), func() *int { y := 1; return &y }(), a.CVector(i2+1-1, (*j1)+i2-1-1), func() *int { y := 1; return &y }())
				}

				//              Swap A(I1, I1) with A(I2, I2)
				piv = a.Get(i1-1, (*j1)+i1-1-1)
				a.Set(i1-1, (*j1)+i1-1-1, a.Get(i2-1, (*j1)+i2-1-1))
				a.Set(i2-1, (*j1)+i2-1-1, piv)

				//              Swap H(I1, I1:J1) with H(I2, I2:J1)
				goblas.Zswap(toPtr(i1-1), h.CVector(i1-1, 0), ldh, h.CVector(i2-1, 0), ldh)
				(*ipiv)[i1-1] = i2

				if i1 > (k1 - 1) {
					//                 Swap L(1:I1-1, I1) with L(1:I1-1, I2),
					//                  skipping the first column
					goblas.Zswap(toPtr(i1-k1+1), a.CVector(i1-1, 0), lda, a.CVector(i2-1, 0), lda)
				}
			} else {
				(*ipiv)[j+1-1] = j + 1
			}

			//           Set A(J+1, J) = T(J+1, J)
			a.Set(j+1-1, k-1, work.Get(1))

			if j < (*nb) {
				//              Copy A(J+1:N, J+1) into H(J+1:N, J),
				goblas.Zcopy(toPtr((*m)-j), a.CVector(j+1-1, k+1-1), func() *int { y := 1; return &y }(), h.CVector(j+1-1, j+1-1), func() *int { y := 1; return &y }())
			}

			//           Compute L(J+2, J+1) = WORK( 3:N ) / T(J, J+1),
			//            where A(J, J+1) = T(J, J+1) and A(J+2:N, J) = L(J+2:N, J+1)
			if j < ((*m) - 1) {
				if a.Get(j+1-1, k-1) != zero {
					alpha = one / a.Get(j+1-1, k-1)
					goblas.Zcopy(toPtr((*m)-j-1), work.Off(2), func() *int { y := 1; return &y }(), a.CVector(j+2-1, k-1), func() *int { y := 1; return &y }())
					goblas.Zscal(toPtr((*m)-j-1), &alpha, a.CVector(j+2-1, k-1), func() *int { y := 1; return &y }())
				} else {
					Zlaset('F', toPtr((*m)-j-1), func() *int { y := 1; return &y }(), &zero, &zero, a.Off(j+2-1, k-1), lda)
				}
			}
		}
		j = j + 1
		goto label30
	label40:
	}
}
