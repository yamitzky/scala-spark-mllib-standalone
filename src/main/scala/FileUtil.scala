import java.nio.charset.Charset
import java.nio.file.{Files, Paths}
import scala.collection.JavaConversions._

/**
 * Created by xd on 2014/10/05.
 */
object FileUtil {
  def load(name: String): Seq[String] = {
    val path = Paths.get("data", name)
    Files.readAllLines(path, Charset.defaultCharset()).filter(_ != "")
  }
}
