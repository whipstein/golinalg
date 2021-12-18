package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpbtrf computes the Cholesky factorization of a complex Hermitian
// positive definite band matrix A.
//
// The factorization has the form
//    A = U**H * U,  if UPLO = 'U', or
//    A = L  * L**H,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular.
func Zpbtrf(uplo mat.MatUplo, n, kd int, ab *mat.CMatrix) (info int, err error) {
	var cone complex128
	var one, zero float64
	var i, i2, i3, ib, ii, j, jj, ldwork, nb, nbmax int

	one = 1.0
	zero = 0.0
	cone = (1.0 + 0.0*1i)
	nbmax = 32
	ldwork = nbmax + 1
	work := cmf(ldwork, 32, opts)

	//     Test the input parameters.
	if (uplo != Upper) && (uplo != Lower) {
		err = fmt.Errorf("(uplo != Upper) && (uplo != Lower): uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	}
	if err != nil {
		gltest.Xerbla2("Zpbtrf", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Determine the block size for this environment
	nb = Ilaenv(1, "Zpbtrf", []byte{uplo.Byte()}, n, kd, -1, -1)

	//     The block size must not exceed the semi-bandwidth KD, and must not
	//     exceed the limit set by the size of the local array WORK.
	nb = min(nb, nbmax)

	if nb <= 1 || nb > kd {
		//        Use unblocked code
		if info, err = Zpbtf2(uplo, n, kd, ab); err != nil {
			panic(err)
		}
	} else {
		//        Use blocked code
		if uplo == Upper {
			//           Compute the Cholesky factorization of a Hermitian band
			//           matrix, given the upper triangle of the matrix in band
			//           storage.
			//
			//           Zero the upper triangle of the work array.
			for j = 1; j <= nb; j++ {
				for i = 1; i <= j-1; i++ {
					work.SetRe(i-1, j-1, zero)
				}
			}

			//           Process the band matrix one diagonal block at a time.
			for i = 1; i <= n; i += nb {
				ib = min(nb, n-i+1)

				//              Factorize the diagonal block
				if ii, err = Zpotf2(uplo, ib, ab.Off(kd, i-1).UpdateRows(ab.Rows-1)); err != nil {
					panic(err)
				}
				if ii != 0 {
					info = i + ii - 1
					return
				}
				if i+ib <= n {
					//                 Update the relevant part of the trailing submatrix.
					//                 If A11 denotes the diagonal block which has just been
					//                 factorized, then we need to update the remaining
					//                 blocks in the diagram:
					//
					//                    A11   A12   A13
					//                          A22   A23
					//                                A33
					//
					//                 The numbers of rows and columns in the partitioning
					//                 are IB, I2, I3 respectively. The blocks A12, A22 and
					//                 A23 are empty if IB = KD. The upper triangle of A13
					//                 lies outside the band.
					i2 = min(kd-ib, n-i-ib+1)
					i3 = min(ib, n-i-kd+1)

					if i2 > 0 {
						//                    Update A12
						if err = ab.Off(kd+1-ib-1, i+ib-1).UpdateRows(ab.Rows-1).Trsm(Left, Upper, ConjTrans, NonUnit, ib, i2, cone, ab.Off(kd, i-1).UpdateRows(ab.Rows-1)); err != nil {
							panic(err)
						}

						//                    Update A22
						if err = ab.Off(kd, i+ib-1).UpdateRows(ab.Rows-1).Herk(Upper, ConjTrans, i2, ib, -one, ab.Off(kd+1-ib-1, i+ib-1).UpdateRows(ab.Rows-1), one); err != nil {
							panic(err)
						}
					}

					if i3 > 0 {
						//                    Copy the lower triangle of A13 into the work array.
						for jj = 1; jj <= i3; jj++ {
							for ii = jj; ii <= ib; ii++ {
								work.Set(ii-1, jj-1, ab.Get(ii-jj, jj+i+kd-1-1))
							}
						}

						//                    Update A13 (in the work array).
						if err = work.Trsm(Left, Upper, ConjTrans, NonUnit, ib, i3, cone, ab.Off(kd, i-1).UpdateRows(ab.Rows-1)); err != nil {
							panic(err)
						}

						//                    Update A23
						if i2 > 0 {
							if err = ab.Off(1+ib-1, i+kd-1).UpdateRows(ab.Rows-1).Gemm(ConjTrans, NoTrans, i2, i3, ib, -cone, ab.Off(kd+1-ib-1, i+ib-1).UpdateRows(ab.Rows-1), work, cone); err != nil {
								panic(err)
							}
						}

						//                    Update A33
						if err = ab.Off(kd, i+kd-1).UpdateRows(ab.Rows-1).Herk(Upper, ConjTrans, i3, ib, -one, work, one); err != nil {
							panic(err)
						}

						//                    Copy the lower triangle of A13 back into place.
						for jj = 1; jj <= i3; jj++ {
							for ii = jj; ii <= ib; ii++ {
								ab.Set(ii-jj, jj+i+kd-1-1, work.Get(ii-1, jj-1))
							}
						}
					}
				}
			}
		} else {
			//           Compute the Cholesky factorization of a Hermitian band
			//           matrix, given the lower triangle of the matrix in band
			//           storage.
			//
			//           Zero the lower triangle of the work array.
			for j = 1; j <= nb; j++ {
				for i = j + 1; i <= nb; i++ {
					work.SetRe(i-1, j-1, zero)
				}
			}

			//           Process the band matrix one diagonal block at a time.
			for i = 1; i <= n; i += nb {
				ib = min(nb, n-i+1)

				//              Factorize the diagonal block
				if ii, err = Zpotf2(uplo, ib, ab.Off(0, i-1).UpdateRows(ab.Rows-1)); err != nil {
					panic(err)
				}
				if ii != 0 {
					info = i + ii - 1
					return
				}
				if i+ib <= n {
					//                 Update the relevant part of the trailing submatrix.
					//                 If A11 denotes the diagonal block which has just been
					//                 factorized, then we need to update the remaining
					//                 blocks in the diagram:
					//
					//                    A11
					//                    A21   A22
					//                    A31   A32   A33
					//
					//                 The numbers of rows and columns in the partitioning
					//                 are IB, I2, I3 respectively. The blocks A21, A22 and
					//                 A32 are empty if IB = KD. The lower triangle of A31
					//                 lies outside the band.
					i2 = min(kd-ib, n-i-ib+1)
					i3 = min(ib, n-i-kd+1)

					if i2 > 0 {
						//                    Update A21
						if err = ab.Off(1+ib-1, i-1).UpdateRows(ab.Rows-1).Trsm(Right, Lower, ConjTrans, NonUnit, i2, ib, cone, ab.Off(0, i-1).UpdateRows(ab.Rows-1)); err != nil {
							panic(err)
						}

						//                    Update A22
						if err = ab.Off(0, i+ib-1).UpdateRows(ab.Rows-1).Herk(Lower, NoTrans, i2, ib, -one, ab.Off(1+ib-1, i-1).UpdateRows(ab.Rows-1), one); err != nil {
							panic(err)
						}
					}

					if i3 > 0 {
						//                    Copy the upper triangle of A31 into the work array.
						for jj = 1; jj <= ib; jj++ {
							for ii = 1; ii <= min(jj, i3); ii++ {
								work.Set(ii-1, jj-1, ab.Get(kd+1-jj+ii-1, jj+i-1-1))
							}
						}

						//                    Update A31 (in the work array).
						if err = work.Trsm(Right, Lower, ConjTrans, NonUnit, i3, ib, cone, ab.Off(0, i-1).UpdateRows(ab.Rows-1)); err != nil {
							panic(err)
						}

						//                    Update A32
						if i2 > 0 {
							if err = ab.Off(1+kd-ib-1, i+ib-1).UpdateRows(ab.Rows-1).Gemm(NoTrans, ConjTrans, i3, i2, ib, -cone, work, ab.Off(1+ib-1, i-1).UpdateRows(ab.Rows-1), cone); err != nil {
								panic(err)
							}
						}

						//                    Update A33
						if err = ab.Off(0, i+kd-1).UpdateRows(ab.Rows-1).Herk(Lower, NoTrans, i3, ib, -one, work, one); err != nil {
							panic(err)
						}

						//                    Copy the upper triangle of A31 back into place.
						for jj = 1; jj <= ib; jj++ {
							for ii = 1; ii <= min(jj, i3); ii++ {
								ab.Set(kd+1-jj+ii-1, jj+i-1-1, work.Get(ii-1, jj-1))
							}
						}
					}
				}
			}
		}
	}

	return
}
